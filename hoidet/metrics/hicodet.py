"""
测试 HOI detection 模型在 HICO-DET 数据集上的性能
"""
import json
import torch
from tqdm import tqdm

from hoidet.dataset import HICO_DET_INFO, DatasetInfo, HICODet
from hoidet.utils import BoxPairAssociation, DetectionAPMeter


class HICODetMetric:
    def __init__(self, pred_file_path: str,
                 partition: str = "test2015",
                 dataset_info: DatasetInfo = HICO_DET_INFO,
                 tp_min_iou: float = 0.5
                 ):
        with open(pred_file_path, mode="r", encoding='utf-8') as fd:
            self.preds = json.load(fd)

        self.hicodet = HICODet(
            dataset_info=dataset_info,
            partition=partition
        )

        # 计算哪些预测结果是 True Positive
        self.tp_associate = BoxPairAssociation(min_iou=tp_min_iou)

        # 计算 AP
        self.meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=self.hicodet.hoi_instance_num,
            algorithm='11P'
        )

        self.ap = self._eval()

    def _eval(self):
        for pred in tqdm(self.preds):
            image_id = pred['image_id']
            boxes = torch.FloatTensor(pred['boxes'])
            boxes_h = boxes[pred['h_idx']]
            boxes_o = boxes[pred['o_idx']]
            hoi_score = torch.FloatTensor(pred['hoi_score'])
            hoi_class = torch.LongTensor(pred['hoi_class'])

            # 根据 image_id 获取 ground truth 标签
            sample_idx = self.hicodet.get_index(image_id)
            target = self.hicodet.annotations[sample_idx]
            gt_boxes_h = torch.FloatTensor(target['boxes_h'])
            gt_boxes_o = torch.FloatTensor(target['boxes_o'])

            # 统计哪些是 TP
            labels = torch.zeros_like(hoi_score)
            unique_hoi = hoi_class.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(torch.LongTensor(target['hoi']) == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(hoi_class == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = self.tp_associate(
                        (gt_boxes_h[gt_idx].view(-1, 4), gt_boxes_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4), boxes_o[det_idx].view(-1, 4)),
                        scores=hoi_score[det_idx].view(-1)
                    )

            self.meter.append(hoi_score, hoi_class, labels)

        return self.meter.eval()

    def get_ap(self):
        return self.ap

    def get_full_map(self):
        return self.ap.mean()

    def get_rare_map(self):
        pass

    def get_non_rare_map(self):
        pass


if __name__ == '__main__':
    hicodet = HICODetMetric(pred_file_path="../../data/hicodet_pred.json")

    print(f"mAP: {hicodet.get_full_map():0.4f}")
