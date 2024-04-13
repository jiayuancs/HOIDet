"""
修改自：https://github.com/fredzzhang/pocket/blob/master/pocket/data/hicodet.py
"""

import os
import json
import numpy as np

from typing import Optional, List, Callable, Tuple
from hoidet.dataset.base import DatasetInfo, DatasetBase
from hoidet.dataset.config import HICO_DET_INFO


class HICODet(DatasetBase):
    """
    Arguments:
        dataset_info(DatasetInfo): 数据集基本信息
        partition(str): 数据集分区
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """

    def __init__(self,
                 partition: str,
                 dataset_info: DatasetInfo = HICO_DET_INFO,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super(HICODet, self).__init__(transform, target_transform, transforms)

        self.name = dataset_info.get_name()
        self.anno_path = dataset_info.get_anno_path(partition)
        self.image_path = dataset_info.get_image_path(partition)

        self.object_class_num = dataset_info.get_object_class_num()
        self.hoi_class_num = dataset_info.get_hoi_class_num()
        self.verb_class_num = dataset_info.get_verb_class_num()

        # 加载并处理数据集
        self._load_annotation_and_metadata()

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "image_id": int
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        return self._transforms(
            self.load_image(os.path.join(self.image_path, self._filenames[i])),
            self._anno[i]
        )

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.object_class_num, self.verb_class_num], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_hoi(self) -> List[list]:
        """
        每个物体类别可能对应的所有HOI类别编号
        """
        obj_to_int = [[] for _ in range(self.object_class_num)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        每个物体类别可能对应的所有动词类别编号
        """
        obj_to_verb = [[] for _ in range(self.object_class_num)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def hoi_instance_num(self) -> List[int]:
        """
        每个HOI类别对应的HOI实例个数
        """
        return self._hoi_instance_num.copy()

    @property
    def object_instance_num(self) -> List[int]:
        """
        每个物体类别对应的HOI实例个数
        """
        num_anno = [0 for _ in range(self.object_class_num)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._hoi_instance_num[corr[0]]
        return num_anno

    @property
    def verb_instance_num(self) -> List[int]:
        """
        每个动词类别对应的HOI实例个数
        """
        num_anno = [0 for _ in range(self.verb_class_num)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._hoi_instance_num[corr[0]]
        return num_anno

    @property
    def hoi_class_names(self) -> List[str]:
        """
        Combination of verbs and objects
        示例：
            HICO-DET数据集中有600种HOI，因此这里返回的就是一个长度为600的列表，
            列表中的每个元素都是字符串，格式为'verb object'，例如：
            - 'wash toothbrush'
            - 'jump snowboard'

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i]
                for _, i, j in self._class_corr]

    def _load_annotation_and_metadata(self) -> None:
        with open(self.anno_path, 'r') as fd:
            f = json.load(fd)

        # 删除不包含任何人物对的样本
        keep = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            keep.remove(empty_idx)

        # 统计每个 HOI 实例的个数（一个图片中可能包含多个实例）
        hoi_instance_num = [0 for _ in range(self.hoi_class_num)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                hoi_instance_num[hoi] += 1
        self._hoi_instance_num = hoi_instance_num

        # 删除不包含任何人物对的样本
        self._anno = [f['annotation'][idx] for idx in keep]
        self._image_sizes = [f['size'][idx] for idx in keep]  # 图片大小
        self._filenames = [f['filenames'][idx] for idx in keep]  # 图片名称

        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']

        # 计算图片ID
        image_ids = [int(img_name.split('_')[-1].split('.')[0]) for img_name in self._filenames]

        # 添加 image_id 字段
        for idx, anno in enumerate(self._anno):
            anno['image_id'] = image_ids[idx]

        # 图片ID到索引的映射
        self._image_id_to_index = dict()
        for index, img_id in enumerate(image_ids):
            self._image_id_to_index[img_id] = index

        # HICO-DET 边界框坐标范围是 [1, W] 或 [1,H]，其中 (W,H) 是图像宽高，
        # 现将其范围变为 [0, W] 或 [0, H]（即zero-based index）
        for i in range(len(self._anno)):
            for j in range(len(self._anno[i]['boxes_h'])):
                self._anno[i]['boxes_h'][j][0] -= 1
                self._anno[i]['boxes_h'][j][1] -= 1
                self._anno[i]['boxes_o'][j][0] -= 1
                self._anno[i]['boxes_o'][j][1] -= 1


if __name__ == '__main__':
    from config import HICO_DET_INFO

    hico_det_train = HICODet(
        dataset_info=HICO_DET_INFO,
        partition="train2015"
    )

    print(hico_det_train)

    from hoidet.visualization import draw_box_pairs

    image, target = hico_det_train[2]
    draw_box_pairs(image, target['boxes_h'], target['boxes_o'], width=3)
    image.save("output_image.jpg")
    print(hico_det_train.get_hoi_class_name(2))
