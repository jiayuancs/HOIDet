import os
import pickle
import numpy as np
from typing import List
from torch import Tensor


from hoidet.metrics.vsrl_eval import VCOCOeval
from hoidet.dataset import VCOCO_INFO, DatasetInfo

__all__ = ['VCOCOMetric', 'VCOCOResultTemplate']


class VCOCOResultTemplate:
    """保存HOI检测模型的预测结果"""

    def __init__(self):
        self.results = []

    def append(self,
               image_id: int,
               boxes: Tensor,
               human_box_idx: Tensor,
               object_box_idx: Tensor,
               hoi_score: Tensor,
               verb_class_names: List):
        """
        添加预测结果
        Args:
            image_id: 图片编号（即图片名称中的数字）
            boxes: 该图片中所有边界框的坐标，格式为[(x1,y1,x2,y2), ...]
            human_box_idx: human_box_idx[i]表示第i个 human-object pair 中，人的边界框在 boxes 中的索引
            object_box_idx: object_box_idx[i]表示第i个 human-object pair 中，物的边界框在 boxes 中的索引
            hoi_score: 表示第 i 个 human-object pair 的置信度分数
            verb_class_names: 表示第 i 个 human-object pair 的**动作**类别名称（注意：不是HOI类别）
        """

        boxes_h = boxes[human_box_idx]
        boxes_o = boxes[object_box_idx]

        for box_h, box_o, score, verb in zip(boxes_h, boxes_o, hoi_score, verb_class_names):
            action, role = verb.split()
            res = dict(image_id=image_id, person_box=box_h.tolist())
            res[f'{action}_agent'] = score.item()
            res[f'{action}_{role}'] = box_o.tolist() + [score.item()]
            self.results.append(res)

    def save(self, file_path):
        """将结果保存到file_path文件中"""
        with open(file_path, mode='wb') as fd:
            pickle.dump(self.results, fd)


class VCOCOMetric:
    def __init__(self, pred_file_path: str,
                 partition: str = "test",
                 dataset_info: DatasetInfo = VCOCO_INFO,
                 tp_min_iou: float = 0.5):
        """
        Args:
            pred_file_path: 模型在partition分区上的预测结果文件路径
            partition: 指定数据集分区
            dataset_info: 数据集基本信息
            tp_min_iou: 只有当人框和物框与真实框IoU的最小值大于等于tp_min_iou，且预测的HOI类别正确时，才可能被认为是 True Positive
        """
        self.pred_file_path = pred_file_path
        self.tp_min_iou = tp_min_iou
        self.partition = partition
        self.evaluation_file_path = os.path.join(
            dataset_info.get_root(),
            dataset_info.get_others()['evaluation_file'][partition]
        )

        self.vcoco_eval = VCOCOeval(
            pred_file_path=self.pred_file_path,
            evaluation_file_path=self.evaluation_file_path
        )

        self.ap1 = None
        self.ap2 = None

    def eval(self):
        self.ap1, self.ap2 = self.vcoco_eval.eval(self.tp_min_iou)

    def get_ap_s1(self):
        return self.ap1

    def get_ap_s2(self):
        return self.ap2

    def get_ap(self, scenario: int):
        assert scenario in [1, 2]
        if scenario == 1:
            return self.get_ap_s1()
        return self.get_ap_s2()

    def get_map_s1(self):
        """
        Note:
            references: https://github.com/fredzzhang/upt/discussions/14
            共计算了 25 个类别的 AP，但是由于类别 point-instr 无有效的人物交互，
            因此，该类别的 AP=0。
            所以，为了得到模型的真实性能，需要将 mAP 乘以 25/24

        Returns: Scenario 1 的 mAP

        """
        mean_ap_s1 = np.nanmean(self.ap1) * 100.00
        mean_ap_s1 = mean_ap_s1 * 25.0 / 24.0
        return mean_ap_s1

    def get_map_s2(self):
        """
        Note:
            references: https://github.com/fredzzhang/upt/discussions/14
            共计算了 25 个类别的 AP，但是由于类别 point-instr 无有效的人物交互，
            因此，该类别的 AP=0。
            所以，为了得到模型的真实性能，需要将 mAP 乘以 25/24

        Returns: Scenario 2 的 mAP

        """
        mean_ap_s2 = np.nanmean(self.ap2) * 100.00
        mean_ap_s2 = mean_ap_s2 * 25.0 / 24.0
        return mean_ap_s2

    def get_map(self, scenario: int):
        assert scenario in [1, 2]
        if scenario == 1:
            return self.get_map_s1()
        return self.get_map_s2()

    def print_map(self, scenario: int):
        self.vcoco_eval.print_role_ap(self.get_ap(scenario), scenario)
        print(f"Remove point-instr: Average Role [scenario_{scenario}] AP = {self.get_map(scenario):.4f}")


if __name__ == "__main__":
    vcoco = VCOCOMetric(pred_file_path="/workspace/code/dl_github/HOIDet/data/vcoco_pred.pkl")
    vcoco.eval()  # 评测
    vcoco.print_map(1)
    vcoco.print_map(2)
