"""
定义目标检测器的通用接口
"""
import torch
import argparse
import torchvision.ops.boxes as box_ops
from torch import nn, Tensor
from typing import List
from argparse import Namespace

__all__ = ['Detector', 'base_detector_args']


def base_detector_args():
    """与目标检测器相关的基础参数"""
    parser = argparse.ArgumentParser(add_help=False)

    # prepare_region_proposals 函数的参数，用于过滤目标检测器输出的边界框
    parser.add_argument('--human_class_id', default=0, type=int,
                        help="目标检测器输出的类别标签中，人这个类别的编号。通常基于COCO数据集训练的目标检测器，人标签编号是0")
    parser.add_argument('--box-score-thresh', default=0.05, type=float,
                        help="过滤掉小于该置信度阈值的边界框")
    parser.add_argument('--min_instance_num', default=3, type=int,
                        help="人框或物框数量的最小数量")
    parser.add_argument('--max_instance_num', default=15, type=int,
                        help="人框或物框数量的最大数量")
    return parser


def prepare_region_proposals(
        results, hidden_states, image_sizes,
        box_score_thresh, human_idx,
        min_instances, max_instances):
    """
    将目标检测器得到的box经过nms处理，并按置信度进行过滤，确保人框/物框的数量都在[min_instances, max_instances]范围内
    Args:
        results: list[dict], 是目标检测器输出结果，其列表长度等于批量大小，每一个字典元素含有三个键：
                scores: 每个框的置信度分数
                labels: 每个框的标签
                boxes: 每个框的坐标
        hidden_states: Tensor(batch_size, query_num, hidden_dim) 是 decoder 输出的每个边界框的特征
        image_sizes: Tensor(batch_size, 2) 每张图片的长宽
        box_score_thresh: float, 过滤掉小于该置信度阈值的边界框
        human_idx: int, 人这个类别对应的编号
        min_instances: int, 如果人框/物框的数量小于min_instances，则将人框/物框按置信度降序排列，忽略box_score_thresh的限制，
                保留置信度最高的min_instances个人框/物框
        max_instances: int, 如果人框/物框的数量打于max_instances，则将人框/物框按置信度降序排列，忽略box_score_thresh的限制，
                保留置信度最高的max_instances个人框/物框

    Returns: list[dict]，其列表长度等于批量大小，每一个字典元素含有四个键：
                boxes: 形状为(box_num, 4), 表示过滤后的所有边界框坐标
                scores: 形状为(box_num,), 表示每个边界框的置信度分数
                labels: 形状为(box_num,), 表示每个边界框的标签编号
                hidden_states: 形状为(box_num, hidden_dim), 表示 decoder 输出的每个边界框的特征
    """
    region_props = []
    for res, hs, sz in zip(results, hidden_states, image_sizes):
        sc, lb, bx = res.values()  # score, label, box

        keep = box_ops.batched_nms(bx, sc, lb, 0.5)
        sc = sc[keep].view(-1)
        lb = lb[keep].view(-1)
        bx = bx[keep].view(-1, 4)
        hs = hs[keep].view(-1, 256)

        # Clamp boxes to image
        bx[:, :2].clamp_(min=0)
        bx[:, 2].clamp_(max=sz[1])
        bx[:, 3].clamp_(max=sz[0])

        keep = torch.nonzero(sc >= box_score_thresh).squeeze(1)

        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human[keep].sum();
        n_object = len(keep) - n_human
        # Keep the number of human and object instances in a specified interval
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            keep_h = keep[keep_h]

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            keep_o = keep[keep_o]

        keep = torch.cat([keep_h, keep_o])

        region_props.append(dict(
            boxes=bx[keep],
            scores=sc[keep],
            labels=lb[keep],
            hidden_states=hs[keep]
        ))

    return region_props


class Detector(nn.Module):
    def __init__(self, args: Namespace):
        """
        目标检测器基类
        Args:
            args: 应包含 base_detector_args 函数中的参数
        """
        super().__init__()
        self.box_score_thresh = args.box_score_thresh
        self.human_class_id = args.human_class_id
        self.min_instance_num = args.min_instance_num
        self.max_instance_num = args.max_instance_num

    def freeze_params(self):
        """冻结目标检测器的所有参数"""
        raise NotImplementedError

    def _base_forward(self, images):
        raise NotImplementedError

    def _postprocessor(self, results, image_sizes):
        raise NotImplementedError

    def _prepare_region_proposals(self, results, hidden_states, image_sizes):
        return prepare_region_proposals(
            results=results,
            hidden_states=hidden_states,
            image_sizes=image_sizes,
            box_score_thresh=self.box_score_thresh,
            human_idx=self.human_class_id,
            min_instances=self.min_instance_num,
            max_instances=self.max_instance_num
        )

    def forward(self, images: List[Tensor], image_sizes: Tensor):
        """
        Args:
            images: 图片
            image_sizes: 图片大小
        Returns: (region_props, features, hs)
            region_props 是预测的边界框信息，类型为list[dict]，其列表长度等于批量大小，每一个字典元素含有四个键：
                    boxes: 形状为(box_num, 4), 表示过滤后的所有边界框坐标
                    scores: 形状为(box_num,), 表示每个边界框的置信度分数
                    labels: 形状为(box_num,), 表示每个边界框的标签编号
                    hidden_states: 形状为(box_num, hidden_dim), 表示 decoder 输出的每个边界框的特征
            features 是 backbone 输出的图片特征
        """
        with torch.no_grad():
            # 基础的 forward 过程
            results, hs, features = self._base_forward(images)
            # 后处理，得到边界框和置信度分数
            results = self._postprocessor(results, image_sizes)
            # 经过 NMS 处理，并按置信度进行过滤，确保人框/物框的数量都在指定范围内
        region_props = self._prepare_region_proposals(results, hs[-1], image_sizes)
        # 返回边界框信息、backbone输出的图片特征、decoder输出的边界框特征
        return region_props, features
