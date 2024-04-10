"""
修改自：https://github.com/fredzzhang/vcoco/blob/main/vcoco.py
"""

import os
import json
import itertools
import numpy as np

from typing import Optional, List, Callable, Tuple, Any, Dict
from hoidet.dataset.base import ImageDataset, DatasetInfo


class VCOCO(ImageDataset):
    """
    V-COCO dataset

    Parameters:
    -----------
    dataset_info(DatasetInfo):
        数据集基本信息
    partition(str):
        数据集分区
    transform: callable
        A function/transform that  takes in an PIL image and returns a transformed version.
    target_transform: callble
        A function/transform that takes in the target and transforms it.
    transforms: callable
        A function/transform that takes input sample and its target as entry and 
        returns a transformed version.
    """

    def __init__(self,
                 dataset_info: DatasetInfo,
                 partition: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super().__init__(dataset_info.get_image_path(partition), transform, target_transform, transforms)

        self._anno_file = dataset_info.get_anno_path(partition)
        with open(self._anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = dataset_info.get_object_class_num()
        self.num_action_cls = dataset_info.get_verb_class_num()  # 24

        # Compute metadata
        self._compute_metatdata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._coco_image_ids)

    def __getitem__(self, i: int) -> Tuple[Any, Any]:
        """
        Parameters:
        -----------
        i: int
            The index to an image.
        
        Returns:
        --------
        image: Any
            Input Image. By default, when relevant transform arguments are None,
            the image is in the form of PIL.Image.
        target: Any
            The annotation associated with the given image. By default, when
            relevant transform arguments are None, the taget is a dict with the
            following keys:
                boxes_h: List[list]
                    Human bouding boxes in a human-object pair encoded as the top
                    left and bottom right corners
                boxes_o: List[list]
                    Object bounding boxes corresponding to the human boxes
                verb: List[int]
                    Ground truth action class for each human-object pair
                object: List[int]
                    Object category index for each object in human-object pairs. The
                    indices follow the 80-class standard, where 0 means background and
                    1 means person.
        """
        image = self.load_image(os.path.join(
            self._root, self.filename(i)
        ))
        target = self._anno[i].copy()
        return self._transforms(image, target)

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._anno_file)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def verbs(self) -> List[str]:
        """Return the list of actions"""
        return self._verbs

    @property
    def objects(self) -> List[str]:
        """Return the list of objects"""
        return self._objects

    @property
    def present_objects(self) -> List[int]:
        """Return the list of objects that are present in the dataset partition"""
        return self._present_objects

    @property
    def verb_instance_num(self) -> List[int]:
        """Return the number of human-object pairs for each action class"""
        return self._verb_instance_num

    @property
    def verb_to_object(self) -> List[list]:
        """Return the list of objects for each action"""
        return self._verb_to_object

    @property
    def object_to_verb(self) -> Dict[int, list]:
        """Return the list of actions for each object"""
        object_to_verb_dict = {obj: [] for obj in list(range(1, len(self._objects)))}
        for verb, obj in enumerate(self._verb_to_object):
            for o in obj:
                if verb not in object_to_verb_dict[o]:
                    object_to_verb_dict[o].append(verb)
        return object_to_verb_dict

    def coco_image_id(self, idx: int) -> int:
        """Return the COCO image ID"""
        return self._coco_image_ids[idx]

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[idx]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_size[idx]

    def _compute_metatdata(self, f: dict) -> None:
        # keep 是符合要求的样本下标列表（即至少包含1个动作的图片下标列表）
        keep = list(range(len(f['annotations'])))
        # action_instance_num[i]表示第i个动作的实例个数（一张图片中可能存在多个实例）
        action_instance_num = [0 for _ in range(len(f['verbs']))]
        # valid_objects[i]表示第i个动作对应的目标类别列表
        valid_objects = [[] for _ in range(len(f['verbs']))]
        for i, anno_in_image in enumerate(f['annotations']):
            # Remove images without human-object pairs
            if len(anno_in_image['verb']) == 0:  # 如果该图片中不存在任何动作，则删除该图片
                keep.remove(i)
                continue
            for act, obj in zip(anno_in_image['verb'], anno_in_image['object']):
                action_instance_num[act] += 1
                if obj not in valid_objects[act]:
                    valid_objects[act].append(obj)

        # self._present_objects 是该分区中实际出现的物体类别编号列表
        objects = list(itertools.chain.from_iterable(valid_objects))
        self._present_objects = np.unique(np.asarray(objects)).tolist()
        # self._action_instance_num[i]表示第i个动作的实例个数（一张图片中可能存在多个实例）
        self._verb_instance_num = action_instance_num

        # 删除不包含任何人物对的样本
        self._anno = [f['annotations'][idx] for idx in keep]     # List[dict]，每张图片的所有标注信息
        self._coco_image_ids = [f['coco_image_id'][idx] for idx in keep]  # 图片在coco数据集中的编号
        self._image_size = [f['size'][idx] for idx in keep]      # 图片大小
        self._filenames = [f['filenames'][idx] for idx in keep]  # 图片名称

        self._verbs = f['verbs']        # 每个动作的名称
        self._objects = f['objects']    # 每个目标的名称
        self._verb_to_object = f['verb_to_object']   # 每个动作可对应的目标列表


if __name__ == '__main__':
    from config import VCOCO_INFO

    vcoco = VCOCO(
        dataset_info=VCOCO_INFO,
        partition="trainval"
    )

    print(vcoco)
