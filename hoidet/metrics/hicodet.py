"""
测试 HOI detection 模型在 HICO-DET 数据集上的性能
"""
import torch
import pickle
from torch import Tensor
from tqdm import tqdm

from hoidet.dataset import HICO_DET_INFO, DatasetInfo, HICODet
from hoidet.utils.association import BoxPairAssociation
from hoidet.utils.meters import DetectionAPMeter

__all__ = ['HICODetMetric', 'HICODetResultTemplate']


class HICODetResultTemplate:
    """保存HOI检测模型的预测结果"""

    def __init__(self):
        self.results = []

    def append(self,
               image_id: int,
               boxes: Tensor,
               human_box_idx: Tensor,
               object_box_idx: Tensor,
               hoi_score: Tensor,
               hoi_class: Tensor):
        """
        添加预测结果
        Args:
            image_id: 图片编号（即图片名称中的数字）
            boxes: 该图片中所有边界框的坐标，格式为[(x1,y1,x2,y2), ...]
            human_box_idx: human_box_idx[i]表示第i个 human-object pair 中，人的边界框在 boxes 中的索引
            object_box_idx: object_box_idx[i]表示第i个 human-object pair 中，物的边界框在 boxes 中的索引
            hoi_score: 表示第 i 个 human-object pair 的置信度分数
            hoi_class: 表示第 i 个 human-object pair 的 HOI 类别编号（注意：不是动词类别编号）
        """
        self.results.append({
            'image_id': image_id,
            'boxes': boxes.tolist(),
            'h_idx': human_box_idx.tolist(),
            'o_idx': object_box_idx.tolist(),
            'hoi_score': hoi_score.tolist(),
            'hoi_class': hoi_class.int().tolist()
        })

    def save(self, file_path):
        """将结果保存到file_path文件中"""
        with open(file_path, mode='wb') as fd:
            pickle.dump(self.results, fd)


class HICODetMetric:
    def __init__(self, pred_file_path: str,
                 partition: str = "test2015",
                 partition_for_rare: str = "train2015",
                 dataset_info: DatasetInfo = HICO_DET_INFO,
                 tp_min_iou: float = 0.5):
        """
        Args:
            pred_file_path: 模型在partition分区上的预测结果文件路径
            partition: 指定要测试的数据集分区
            partition_for_rare: 指定用于统计rare和non-rare类别的数据集分区
            dataset_info: 数据集基本信息
            tp_min_iou: 当人框和物框与真实框IoU的最小值大于tp_min_iou，且预测的HOI类别正确时，被认为是 True Positive
        """
        self.tp_min_iou = tp_min_iou

        with open(pred_file_path, "rb") as fd:
            self.preds = pickle.load(fd)

        self.hicodet = HICODet(
            dataset_info=dataset_info,
            partition=partition
        )
        self.hicodet_for_rare = HICODet(
            dataset_info=dataset_info,
            partition=partition_for_rare
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

    def get_rare_map(self, hoi_class_num_threshold=10):
        """
        Args:
            hoi_class_num_threshold: HOI类别实例个数阈值
                小于该阈值的 HOI 类别被判定为 rare 类别
                大于等于该阈值的 HOI 类别被判定为 non-rare 类别

        Returns: float
            返回 rare 类别的 mAP
        """
        num_anno = torch.as_tensor(self.hicodet_for_rare.hoi_instance_num)
        rare = torch.nonzero(num_anno < hoi_class_num_threshold).squeeze(1)
        return self.ap[rare].mean()

    def get_non_rare_map(self, hoi_class_num_threshold=10):
        """
        Args:
            hoi_class_num_threshold: HOI类别实例个数阈值
                小于该阈值的 HOI 类别被判定为 rare 类别
                大于等于该阈值的 HOI 类别被判定为 non-rare 类别

        Returns: float
            返回 non-rare 类别的 mAP
        """
        num_anno = torch.as_tensor(self.hicodet_for_rare.hoi_instance_num)
        non_rare = torch.nonzero(num_anno >= hoi_class_num_threshold).squeeze(1)
        return self.ap[non_rare].mean()

    def summary(self):
        return (f"mAP: {self.get_full_map():.4f},\t"
                f"rare mAP: {self.get_rare_map():.4f},\t"
                f"non-rare mAP: {self.get_non_rare_map():.4f}")


if __name__ == '__main__':
    hicodet = HICODetMetric(pred_file_path="/workspace/code/dl_github/HOIDet/data/hicodet_pred.pkl")
    hicodet.eval()
    print(hicodet.summary())
