"""
测试 HOI detection 模型在 HICO-DET 数据集上的性能
"""
import json
import torch
from tqdm import tqdm

from hoidet.dataset import HICO_DET_INFO, DatasetInfo, HICODet
from hoidet.utils import BoxPairAssociation, DetectionAPMeter

__all__ = ['HICODetMetric']


class HICODetMetric:
    def __init__(self, pred_file_path: str,
                 partition: str = "test2015",
                 dataset_info: DatasetInfo = HICO_DET_INFO,
                 tp_min_iou: float = 0.5):
        """
        Args:
            pred_file_path: 模型在partition分区上的预测结果文件路径
            partition: 指定数据集分区
            dataset_info: 数据集基本信息
            tp_min_iou: 当人框和物框与真实框IoU的最小值大于tp_min_iou，且预测的HOI类别正确时，被认为是 True Positive
        """
        self.tp_min_iou = tp_min_iou

        with open(pred_file_path, mode="r", encoding='utf-8') as fd:
            self.preds = json.load(fd)

        self.hicodet = HICODet(
            dataset_info=dataset_info,
            partition=partition
        )

        self.ap = None

    def eval(self):
        # 计算哪些预测结果是 True Positive
        tp_associate = BoxPairAssociation(min_iou=self.tp_min_iou)

        # 计算 AP
        meter = DetectionAPMeter(
            self.hicodet.hoi_class_num, nproc=1,
            num_gt=self.hicodet.hoi_instance_num,
            algorithm='11P'
        )

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
                    labels[det_idx] = tp_associate(
                        (gt_boxes_h[gt_idx].view(-1, 4), gt_boxes_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4), boxes_o[det_idx].view(-1, 4)),
                        scores=hoi_score[det_idx].view(-1)
                    )

            meter.append(hoi_score, hoi_class, labels)

        self.ap = meter.eval()

    def get_ap(self):
        return self.ap

    def get_full_map(self):
        return self.ap.mean()

    def get_rare_map(self):
        pass

    def get_non_rare_map(self):
        pass


if __name__ == '__main__':
    hicodet = HICODetMetric(pred_file_path="/workspace/code/dl_github/HOIDet/data/hicodet_pred.json")
    hicodet.eval()
    print(f"mAP: {hicodet.get_full_map():0.4f}")
