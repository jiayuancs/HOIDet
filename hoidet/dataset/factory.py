from torch.utils.data import Dataset

from hoidet.dataset.base import DatasetInfo
from hoidet.dataset.hicodet import HICODet
from hoidet.dataset.vcoco import VCOCO
from hoidet.dataset.config import HICO_DET_INFO, VCOCO_INFO
from hoidet.utils.transforms import to_dict_of_tensor
from hoidet.utils import transforms as T

__all__ = ['custom_collate', 'DataFactory']

# 数据集与相应的数据类的对应关系
DATASET_MAP = {
    HICO_DET_INFO.get_name(): HICODet,
    VCOCO_INFO.get_name(): VCOCO
}


def custom_collate(batch):
    """将批量数据中的图片数据和标签数据分开存放"""
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets


class DataFactory(Dataset):
    def __init__(self, partition: str, dataset_info: DatasetInfo):
        """
        Args:
            partition: 数据集分区
                如果 partition==dataset_info.get_training_partition(), 则会添加额外的数据变换;
                如果 partition!=dataset_info.get_training_partition(), 则仅对图片进行resize和normalize操作.
            dataset_info: 数据集基本信息
        """
        assert dataset_info.get_name() in DATASET_MAP.keys()
        dataset_type = DATASET_MAP[dataset_info.get_name()]

        self.name = dataset_info.get_name()
        self.is_training = partition == dataset_info.get_training_partition()
        self.dataset = dataset_type(
            partition=partition,
            dataset_info=dataset_info,
            target_transform=to_dict_of_tensor
        )

        # 数据变换, 摘自: https://github.com/fredzzhang/upt/blob/main/utils.py
        normalize = T.Compose([
            # T.ToTensor(),  # 默认情况下 self.dataset 中已经执行了 ToTensor()
            # TODO: 统计所有深度图像的均值和方差
            T.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.2])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if self.is_training:  # 如果是训练集，则添加额外的数据变换
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
            ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        # TODO: 修改 hoidet.utils.transforms 中的变换方法，使用 verb 字段而不是 labels 字段
        target['labels'] = target['verb']
        image, target = self.transforms(image, target)
        return image, target

    def __str__(self) -> str:
        return str(self.dataset) + f"Training: {self.is_training}\n"


if __name__ == '__main__':
    import torch
    from PIL import Image


    def _scale_to_255(x: torch.Tensor):
        # 将张量的最小值移动到非负区间
        min_val = x.min()
        x = x - min_val

        # 将张量的最大值移动到[0, 255]范围内
        max_val = x.max()
        if max_val != 0:
            x = x * (255 / max_val)

        # 将张量四舍五入为整数
        x = torch.round(x)

        return x.to(torch.uint8)

    hicodet_trainset = DataFactory(
        partition=HICO_DET_INFO.get_training_partition(),
        dataset_info=HICO_DET_INFO
    )
    # vcoco_trainset = DataFactory(
    #     partition=VCOCO_INFO.get_training_partition(),
    #     dataset_info=VCOCO_INFO
    # )
    # print(hicodet_trainset[0])
    # print(vcoco_trainset[1])

    idx = hicodet_trainset.dataset.get_index(19135)
    rgbd_image, target = hicodet_trainset[idx]
    image = rgbd_image[:3]
    depth = rgbd_image[3]

    # 将张量的值缩放到0到255之间，并转换为整数
    image = _scale_to_255(image)
    depth = _scale_to_255(depth)

    # 将张量转换为PIL图像
    pil_image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
    pil_depth = Image.fromarray(depth.cpu().numpy(), mode='L')
    pil_image.save("output_image.jpg")
    pil_depth.save("output_image_depth.jpg")
    print(hicodet_trainset.dataset.get_hoi_class_name(2))
