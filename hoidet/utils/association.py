"""
Data association in detection tasks

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch

from typing import Tuple, Optional
from torch import FloatTensor, LongTensor

from hoidet.utils.boxes import box_iou


class BoxAssociation:
    """
    Associate detection boxes with ground truth boxes
    用于object detection任务，根据IoU和边界框的置信度分数来判断哪些样本是true positive

    Arguments:
        min_iou(float): 最小IoU阈值（小于等于该值的框被丢弃）
        encoding(str): 默认coord就完事了
    """

    def __init__(self, min_iou: float, encoding: str = 'coord') -> None:
        self.min_iou = min_iou
        self.encoding = encoding

        self._max_iou = None
        self._max_idx = None

    @property
    def max_iou(self) -> FloatTensor:
        """Return the largest IoU with any ground truth instances for each detection"""
        if self._max_iou is None:
            raise NotImplementedError
        else:
            return self._max_iou

    @property
    def max_idx(self) -> LongTensor:
        """Return the index of ground truth instance each detection is associated with"""
        if self._max_idx is None:
            raise NotImplementedError
        else:
            return self._max_idx

    def _iou(self, boxes_1: FloatTensor, boxes_2: FloatTensor) -> FloatTensor:
        """Compute intersection over union"""
        return box_iou(boxes_1, boxes_2, encoding=self.encoding)

    def __call__(self,
                 gt_boxes: FloatTensor,
                 det_boxes: FloatTensor,
                 scores: Optional[FloatTensor] = None
                 ) -> FloatTensor:
        """
        本方法是针对特定的类别而言的，对于给定的HOI类别c，
            - gt_boxes是所有属于c类别的边界框（ground truth）
            - det_boxes是所有预测为c类别的边界框
            - scores是每个预测框被预测为c类别的置信度

        [计算方法]
        首先，对每个预测框det_box:
            1. 计算det_box与所有真实框(gt_boxes)的IoU
            2. 将每个预测框分配给与其IoU最大的真实框，即每个预测框仅与一个真实框匹配，但每个真实框可能匹配了多个预测框
            3. 过滤掉所有小于等于self.min_iou的匹配
        然后，对于每个真实框gt_box:
            1. 如果存在与之匹配的预测框，则仅保留置信度最大score的那个预测框
        经过上述两个步骤的筛选，保留下来的预测框即为true positive，那些被过滤掉的预测框即为false positive.

        Arguments:
            gt_boxes(FloatTensor[N, 4]): Ground truth bounding boxes in (x1, y1, x2, y2) format
            det_boxes(FloatTensor[M, 4]): Detected bounding boxes in (x1, y1, x2, y2) format
            scores(FloatTensor[M]): Confidence scores for each detection. If left as None, the
                highest IoU will be used to rank duplicated detections.
        Returns:
            labels(FloatTensor[M]): Binary labels indicating true positive or not
                形状为[M,]，其中第i个元素为1表示det_boxes[i]是true positive。
                true positive预测框的个数小于等于真实边界框的个数。
        """
        # iou的形状是[N, M], iou[i, j]表示第i个真实框与第j个预测框之间的IoU
        iou = self._iou(gt_boxes, det_boxes)

        # 将每个预测框分给与其具有最大IoU的真实框
        max_iou, max_idx = iou.max(0)
        self._max_iou = max_iou  # self._max_iou[i]表示第i个预测框与真实框的最大IoU值
        self._max_idx = max_idx  # self._max_idx[i]表示与第i个预测框具有最大IoU值的真实框编号

        if scores is None:
            scores = max_iou

        # match[N,M]每一列最多只有一个非0值，该值对应预测框与真实框的IoU
        # 实际上，match就是仅保留iou每列中的最大元素，其他元素都置零
        match = torch.zeros_like(iou)
        match[max_idx, torch.arange(iou.shape[1])] = max_iou

        # 过滤掉小于等于最小IoU阈值的框
        match = match > self.min_iou

        labels = torch.zeros_like(scores)
        # Determine true positives
        for i, m in enumerate(match):
            match_idx = torch.nonzero(m).squeeze(1)  # 与第i个真实框匹配的预测框的编号列表
            if len(match_idx) == 0:
                continue
            # 可能有多个预测框匹配到了同一个真实框，这里仅保留置信度(score)最大的那个预测框，
            # 如果有多个置信度都相同且最大，则选择靠前的那个。
            match_scores = scores[match_idx]
            labels[match_idx[match_scores.argmax()]] = 1

        return labels


class BoxPairAssociation(BoxAssociation):
    """
    Associate detection box pairs with ground truth box pairs
    用于HOI detection任务，根据IoU和边界框的置信度分数来判断那些样本是true positive

    Arguments:
        min_iou(float): The minimum intersection over union to identify a positive
        encoding(str): Encodings of the bounding boxes. Choose between 'coord' and 'pixel'
    """

    def __init__(self, min_iou: float, encoding: str = 'coord') -> None:
        super().__init__(min_iou, encoding)

    def _iou(self,
             boxes_1: Tuple[FloatTensor, FloatTensor],
             boxes_2: Tuple[FloatTensor, FloatTensor]) -> FloatTensor:
        """
        Override method to compute IoU for box pairs

        Arguments:
            boxes_1(tuple): Ground truth box pairs(人框、物框)
            boxes_2(tuple): Detection box pairs(人框、物框)
        """
        return torch.min(  # 即以人框和物框中最小的那个IoU为准
            box_iou(boxes_1[0], boxes_2[0], encoding=self.encoding),
            box_iou(boxes_1[1], boxes_2[1], encoding=self.encoding)
        )


if __name__ == '__main__':
    det_boxes = torch.Tensor([[0., 0., 2., 2.], [1, 3, 2, 3], [0, 1, 1, 2], [0, 1, 3, 3]])
    gt_boxes = torch.Tensor([[1., 1., 3., 3.], [0, 0, 1, 2]])
    box_association = BoxAssociation(0.2)
    label = box_association(gt_boxes, det_boxes)
