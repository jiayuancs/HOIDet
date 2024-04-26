"""
训练HOI模型
"""
import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from argparse import Namespace
import torch.distributed as dist

from hoidet.utils.relocate import relocate_to_cpu, relocate_to_cuda
from hoidet.core.distributed import DistributedLearningEngine
from hoidet.metrics import HICODetResultTemplate, VCOCOResultTemplate

__all__ = ['HOIEngine']


def _gen_cache_path(cache_root, prefix="pred"):
    """生成存储cache文件的目录名称"""
    max_number = 0
    for folder_name in os.listdir(cache_root):
        # 目录格式为 prefix_id
        name_parts = folder_name.split("_")
        if len(name_parts) != 2:
            continue

        folder_prefix, folder_id = name_parts
        if not (folder_prefix == prefix and folder_id.isdigit()):
            continue

        max_number = max(max_number, int(folder_id))
    next_number = max_number + 1
    return os.path.join(cache_root, f"{prefix}_{next_number:02d}")


class HOIEngine(DistributedLearningEngine):
    def __init__(self, net, train_dataloader, test_dataloader, config: Namespace):
        super().__init__(
            net, None, train_dataloader,
            print_interval=config.print_interval,
            cache_dir=config.output_dir,
            find_unused_parameters=True
        )
        self.config = config
        self.max_norm = config.clip_max_norm
        self.test_dataloader = test_dataloader

        self._cache_test_result = {
            "hicodet": self._cache_hicodet,
            "vcoco": self._cache_vcoco
        }[test_dataloader.dataset.name]

        self.cache_path = None
        if self._rank == 0:
            # 创建文件夹以保存预测结果
            os.makedirs(config.cache_root, exist_ok=True)
            self.cache_path = _gen_cache_path(config.cache_root)
            os.makedirs(self.cache_path, exist_ok=True)

            # 保存本次训练使用的参数
            with open(os.path.join(self.cache_path, "config.json"), "w", encoding="utf-8") as fd:
                json.dump(vars(config), fd, indent=4)

            print(f"cache dir: {self.cache_path}")

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['cls_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def _on_start(self):
        """训练前保存模型在测试集上的推理结果"""
        self.cache_results()

    def _on_end_epoch(self):
        """每个epoch后保存模型在测试集上的推理结果"""
        self.cache_results()

    @torch.no_grad()
    def cache_results(self):
        """保存模型在测试集上的推理结果"""
        if self._rank == 0:
            print("start testing")
        self._cache_test_result()

    @torch.no_grad()
    def _cache_hicodet(self):
        # TODO: 当批量大小和GPU数量不同时，得到的结果会有细微的差别，暂不知原因
        # 该代码修改自 PVIC，经测试 PVIC 原仓库也存在这种问题
        dataloader = self.test_dataloader
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        all_results = []
        for batch in tqdm(dataloader, disable=(self._rank != 0)):
            inputs = relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]

            results = HICODetResultTemplate()
            for output, target in zip(outputs, targets):
                boxes = output['boxes']
                verbs = output['labels']
                objects = output['objects']
                interactions = conversion[objects, verbs]

                # 模型在推理时会对图片进行resize操作，故这里需要将预测的边界框进行等比缩放
                image_id = target['image_id'].item()
                data_idx = dataset.get_index(image_id)
                ow, oh = dataset.image_sizes[data_idx]
                h, w = output['size']
                scale_fct = torch.as_tensor([
                    ow / w, oh / h, ow / w, oh / h
                ]).unsqueeze(0)
                boxes *= scale_fct

                # TODO: 这里是否要对boxes的坐标减1（UPT和PVIC）

                # 要求output['pairing']的形状为(pair_num, 2)
                results.append(
                    image_id=image_id,
                    boxes=boxes,
                    human_box_idx=output['pairing'][:, 0],
                    object_box_idx=output['pairing'][:, 1],
                    hoi_score=output['scores'],
                    hoi_class=interactions
                )

            # 将预测结果gather到0号进程上
            results_ddp = [None for _ in range(dist.get_world_size())]
            dist.gather_object(
                obj=results.results,
                object_gather_list=results_ddp if self._rank == 0 else None,
                dst=0
            )

            if self._rank == 0:
                for res in results_ddp:
                    all_results.extend(res)

        if self._rank == 0:
            save_path = os.path.join(self.cache_path, f"{dataset.name}_{self._state.epoch:02d}.pkl")
            with open(save_path, mode='wb') as fd:
                pickle.dump(all_results, fd)

    @torch.no_grad()
    def _cache_vcoco(self):
        raise NotImplementedError
