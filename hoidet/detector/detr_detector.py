"""
封装 DETR 目标检测器
"""
import torch
import argparse
from argparse import Namespace

from hoidet.detector.base import Detector, base_detector_args
from detr.models import build_model
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list

__all__ = ['detr_detector_args', 'DETR']


def detr_detector_args():
    """
    Arguments for building the base detector DETR
    摘自：https://github.com/fredzzhang/pvic/blob/main/configs.py
    """
    parser = base_detector_args()
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    # Transformer
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    # Training
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--lr-drop', default=20, type=int)
    parser.add_argument('--lr-drop-factor', default=.2, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--clip-max-norm', default=.1, type=float)

    # Loss
    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Misc.
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--pretrained', default='', help='预训练参数文件路径')
    parser.add_argument('--print-interval', default=100, type=int)
    return parser


class DETR(Detector):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.detector, _, postprocessors = build_model(args)
        self.detector.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
        self.postprocessors = postprocessors['bbox']

    def freeze_params(self):
        for p in self.detector.parameters():
            p.requires_grad = False

    def _base_forward(self, images: NestedTensor):
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        features, pos = self.detector.backbone(images)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])[
            0]

        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out, hs, features

    def _postprocessor(self, results, image_sizes):
        return self.postprocessors(results, image_sizes)


if __name__ == '__main__':
    from hoidet.dataset import DataFactory, HICO_DET_INFO, custom_collate
    from torch.utils.data import DataLoader
    from hoidet.visualization import draw_boxes_with_txt

    # DETR 的参数
    parser = argparse.ArgumentParser(parents=[detr_detector_args(), ])
    args = parser.parse_args()
    args.pretrained = "../../data/detr-r50-hicodet.pth"
    print(args)

    # 实例化目标检测器
    detr = DETR(args)
    detr.freeze_params()

    # 加载数据集
    hicodet = DataFactory(HICO_DET_INFO.get_testing_partition(), HICO_DET_INFO)
    dataloader = DataLoader(
        dataset=hicodet,
        collate_fn=custom_collate, batch_size=1,
        num_workers=1, pin_memory=True, drop_last=True,
        shuffle=False
    )

    for images, targets in dataloader:
        # 获取图片大小
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)

        # 进行目标检测
        region_props, features = detr(images, image_sizes)

        idx = 0  # 对第 idx 张样本进行可视化
        image_id = targets[idx]['image_id'].item()  # 获取图片ID
        index = hicodet.dataset.get_index(image_id)  # 根据图片ID获取样本在数据集中的索引

        # 对预测的边界框进行缩放
        ow, oh = hicodet.dataset.image_sizes[index]  # 真实图片大小
        h, w = image_sizes[idx]  # 数据变换后的图片大小
        scale_fct = torch.as_tensor([
            ow / w, oh / h, ow / w, oh / h
        ]).unsqueeze(0)
        boxes = region_props[idx]['boxes']
        boxes *= scale_fct  # 缩放

        # 获取原始图片，并绘制缩放后的边界框
        image, _ = hicodet.dataset[index]
        label_txt = [hicodet.dataset.objects[i] for i in region_props[idx]['labels']]
        draw_boxes_with_txt(image, boxes, label_txt)
        image.save("output_image.jpg")

        print("saved")
        break
