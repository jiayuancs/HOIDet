import os
import numpy as np

from hoidet.metrics.vsrl_eval import VCOCOeval
from hoidet.dataset import VCOCO_INFO, DatasetInfo

__all__ = ['VCOCOMetric']


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
        print(f"Remove point-instr: Average Role [scenario_{scenario}] AP = {self.get_map(scenario)}")


if __name__ == "__main__":
    vcoco = VCOCOMetric(pred_file_path="/workspace/code/dl_github/HOIDet/data/cache.pkl")
    vcoco.eval()  # 评测
    vcoco.print_map(1)
    vcoco.print_map(2)
