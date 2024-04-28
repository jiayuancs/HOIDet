"""
修改自：https://github.com/fredzzhang/pocket/blob/master/pocket/utils/visual.py
"""

import torch
import numpy as np

from PIL import ImageDraw

__all__ = ['draw_boxes', 'draw_boxes_with_txt', 'draw_box_pairs']


def draw_boxes(image, boxes, **kwargs):
    """
    在image上绘制矩形框

    Arguments:
        image(PIL Image)
        boxes(torch.Tensor[N,4] or np.ndarray[N,4] or List[List[4]]): Bounding box
            coordinates in the format (x1, y1, x2, y2)
        kwargs(dict): Parameters for PIL.ImageDraw.Draw.rectangle
    """
    if isinstance(boxes, (torch.Tensor, list)):
        boxes = np.asarray(boxes)
    elif not isinstance(boxes, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    boxes = boxes.reshape(-1, 4).tolist()

    canvas = ImageDraw.Draw(image)
    for box in boxes:
        canvas.rectangle(box, **kwargs)


def draw_boxes_with_txt(image, boxes, labels, **kwargs):
    """
    在image上绘制矩形框

    Arguments:
        image(PIL Image)
        boxes(torch.Tensor[N,4] or np.ndarray[N,4] or List[List[4]]): Bounding box
            coordinates in the format (x1, y1, x2, y2)
        kwargs(dict): Parameters for PIL.ImageDraw.Draw.rectangle
    """
    if isinstance(boxes, (torch.Tensor, list)):
        boxes = np.asarray(boxes)
    elif not isinstance(boxes, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    boxes = boxes.reshape(-1, 4).tolist()

    canvas = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        canvas.rectangle(box, **kwargs)
        canvas.text((box[0], box[1]), str(label))


def draw_box_pairs(image, boxes_1, boxes_2, width=3):
    """
    boxes_1中的边界框将被绘制成蓝色，boxes_2中的边界框将被绘制成绿色，每对边界框之间使用红线连接.

    Arguments:
        image: PIL图片类型
        boxes_1: (torch.Tensor[N,4] or np.ndarray[N,4] or List[List[4]]): 坐标格式为(x1, y1, x2, y2)
        boxes_2: Same format as above
        width: 边界框线的宽度
    """
    if isinstance(boxes_1, (torch.Tensor, list)):
        boxes_1 = np.asarray(boxes_1)
    elif not isinstance(boxes_1, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    if isinstance(boxes_2, (torch.Tensor, list)):
        boxes_2 = np.asarray(boxes_2)
    elif not isinstance(boxes_2, np.ndarray):
        raise TypeError("Bounding box coords. should be torch.Tensor, np.ndarray or list")
    boxes_1 = boxes_1.reshape(-1, 4)
    boxes_2 = boxes_2.reshape(-1, 4)

    canvas = ImageDraw.Draw(image)
    assert len(boxes_1) == len(boxes_2), "Number of boxes does not match between two given groups"
    for b1, b2 in zip(boxes_1, boxes_2):
        canvas.rectangle(b1.tolist(), outline='#007CFF', width=width)
        canvas.rectangle(b2.tolist(), outline='#46FF00', width=width)
        b_h_centre = (b1[:2] + b1[2:]) / 2
        b_o_centre = (b2[:2] + b2[2:]) / 2
        canvas.line(
            b_h_centre.tolist() + b_o_centre.tolist(),
            fill='#FF4444', width=width
        )
        canvas.ellipse(
            (b_h_centre - width).tolist() + (b_h_centre + width).tolist(),
            fill='#FF4444'
        )
        canvas.ellipse(
            (b_o_centre - width).tolist() + (b_o_centre + width).tolist(),
            fill='#FF4444'
        )


if __name__ == '__main__':
    from hoidet.dataset import HICO_DET_INFO, HICODet

    hicodet = HICODet(partition="train2015")
    idx = 267
    image, target = hicodet[idx]
    # draw_box_pairs(image, target['boxes_h'], target['boxes_o'])
    draw_boxes_with_txt(image, target['boxes_o'], ["human" for i in range(len(target['boxes_h']))])
    image.save("output_image.jpg")
    print(hicodet.get_hoi_class_name(idx))

