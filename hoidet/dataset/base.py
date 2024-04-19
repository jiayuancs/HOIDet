"""
数据集的基本信息：
    - 数据集名称
    - 存储路径
    - 数据集划分情况（train/val/test）
    - 数据集元数据
"""
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple

__all__ = ['DatasetInfo', 'DatasetBase', 'DataDict', 'ImageDataset', 'DataSubset']


class DatasetInfo:
    def __init__(self,
                 name: str,
                 data_root: str,
                 partition_image: dict,
                 partition_anno: dict,
                 training_partition: str,
                 testing_partition: str,
                 object_class_num: int,
                 hoi_class_num: int,
                 verb_class_num: int,
                 others = None):
        """
        描述一个数据集
        Args:
            name: 数据集名称
            data_root: 数据集根目录
            partition_image: 分区名称与图片目录的映射关系
                例如 {"train2015": "hico_20160224_det/images/train2015"}
            partition_anno: 分区名称与标注文件的映射关系
                例如 {"train2015", "instances_train2015.json"}
            training_partition: 用于训练的分区名称,
            testing_partition: 用于测试的分区名称,
            object_class_num: 物体类别数量
            hoi_class_num: HOI类别数量，即动词与物体的组合数量
            verb_class_num: 动词类别数量
            others: 其他自定义数据
        """
        self._name = name
        self._data_root = data_root
        self._partition_image = partition_image
        self._partition_anno = partition_anno
        assert self._partition_image.keys() == self._partition_anno.keys()
        self._partition = list(self._partition_image.keys())

        self._training_partition = training_partition
        self._testing_partition = testing_partition

        self._object_class_num = object_class_num
        self._hoi_class_num = hoi_class_num
        self._verb_class_num = verb_class_num

        self._others = others

    def set_root(self, root: str):
        """
        更新数据集根目录
        Args:
            root: 数据集根目录

        Returns:

        """
        self._data_root = root

    def get_name(self) -> str:
        return self._name

    def get_root(self) -> str:
        return self._data_root

    def get_hoi_class_num(self):
        return self._hoi_class_num

    def get_verb_class_num(self):
        return self._verb_class_num

    def get_object_class_num(self):
        return self._object_class_num

    def get_partitions(self):
        return self._partition

    def get_training_partition(self):
        return self._training_partition

    def get_testing_partition(self):
        return self._testing_partition

    def get_others(self):
        return self._others

    def get_anno_path(self, partition: str) -> str:
        """
        获取指定分区的注释文件完整路径
        Args:
            partition: 分区名称

        Returns:

        """
        assert partition in self._partition, \
            f"Unknown {self._name} partition :{partition}"
        return os.path.join(self._data_root, self._partition_anno[partition])

    def get_image_path(self, partition: str) -> str:
        """
        获取指定分区的图片目录完整路径
        Args:
            partition: 分区名称

        Returns:

        """
        assert partition in self._partition, \
            f"Unknown {self._name} partition :{partition}"
        return os.path.join(self._data_root, self._partition_image[partition])

    def __str__(self) -> str:
        res = (f"Dataset: {self.get_name()}\n"
               f"\troot: {self.get_root()}\n"
               f"\tpartition: {self._partition}\n"
               f"\tobject_class_num: {self._object_class_num}\n"
               f"\tverb_class_num: {self._verb_class_num}\n"
               f"\thoi_class_num: {self._hoi_class_num}\n")
        return res


class DatasetBase(Dataset):
    """
    所有 HOI 数据集的基类

    方法命名规范如下：
        - get_xxx：所有以get开头的方法均接受一个 idx 参数，表示获取第 idx 样本的属性
    """

    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        self._transform = transform
        self._target_transform = target_transform
        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

        self.name = ""
        self.image_path = ""
        self.anno_path = ""

        self.object_class_num = -1
        self.verb_class_num = -1
        self.hoi_class_num = -1

        self._anno = None
        self._verbs = None
        self._objects = None
        self._image_sizes = None
        self._filenames = None

        self._image_id_to_index = dict()

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, i):
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self.image_path)
        reprstr += ', anno_file='
        reprstr += repr(self.anno_path)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        return (f"Dataset: {self.name}\n"
                f"\tNumber of images: {self.__len__()}\n"
                f"\tImage directory: {self.image_path}\n"
                f"\tAnnotation file: {self.anno_path}\n")

    def _load_annotation_and_metadata(self):
        """加载数据集"""
        raise NotImplementedError

    @property
    def annotations(self) -> List[dict]:
        """返回所有图片的标注信息"""
        return self._anno

    @property
    def verbs(self) -> List[str]:
        """返回所有动作名称列表"""
        return self._verbs

    @property
    def objects(self) -> List[str]:
        """返回所有物体类别名称列表"""
        return self._objects

    @property
    def image_sizes(self) -> Tuple[int, int]:
        """返回所有图片大小列表，格式为[(W,H),...]"""
        return self._image_sizes

    @property
    def filenames(self) -> List[str]:
        """返回所有图片名称列表"""
        return self._filenames

    def get_image_id(self, idx: int) -> int:
        """返回指定索引图片的ID（即图片名称中的数字）"""
        return self._anno[idx]['image_id']

    def get_index(self, image_id: int) -> int:
        """获取指定图片ID在该数据集中的索引，返回-1表示该图片ID不存在"""
        return self._image_id_to_index[image_id] if image_id in self._image_id_to_index.keys() else -1

    def get_hoi_class_name(self, idx: int) -> List[str]:
        """返回第idx个图片中所有的HOI文本标签"""
        hoi_name_list = []
        for verb, obj in zip(self._anno[idx]['verb'], self._anno[idx]['object']):
            hoi_name_list.append(f"{self._verbs[verb]} {self._objects[obj]}")
        return hoi_name_list

    @staticmethod
    def load_image(path: str) -> Image:
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')



"""
以下代码摘自: https://github.com/fredzzhang/pocket/blob/master/pocket/data/base.py

Dataset base classes

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


class DataDict(dict):
    r"""
    Data dictionary class. This is a class based on python dict, with
    augmented utility for loading and saving

    Arguments:
        input_dict(dict, optional): A Python dictionary
        kwargs: Keyworded arguments to be stored in the dict

    """

    def __init__(self, input_dict: Optional[dict] = None, **kwargs) -> None:
        data_dict = dict() if input_dict is None else input_dict
        data_dict.update(kwargs)
        super(DataDict, self).__init__(**data_dict)

    def __getattr__(self, name: str) -> Any:
        """Get attribute"""
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute"""
        self[name] = value

    def save(self, path: str, mode: str = 'wb', **kwargs) -> None:
        """Save the dict into a pickle file"""
        with open(path, mode) as f:
            pickle.dump(self.copy(), f, **kwargs)

    def load(self, path: str, mode: str = 'rb', **kwargs) -> None:
        """Load a dict or DataDict from pickle file"""
        with open(path, mode) as f:
            data_dict = pickle.load(f, **kwargs)
        for name in data_dict:
            self[name] = data_dict[name]

    def is_empty(self) -> bool:
        return not bool(len(self))


class StandardTransform:
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py"""

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, inputs: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


class ImageDataset(Dataset):
    """
    Base class for image dataset

    Arguments:
        root(str): Root directory where images are downloaded to
        transform(callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path: str) -> Image:
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')


class DataSubset(Dataset):
    """
    A subset of data with access to all attributes of original dataset

    Arguments:
        dataset(Dataset): Original dataset
        pool(List[int]): The pool of indices for the subset
    """

    def __init__(self, dataset: Dataset, pool: List[int]) -> None:
        self.dataset = dataset
        self.pool = pool

    def __len__(self) -> int:
        return len(self.pool)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.pool[idx]]

    def __getattr__(self, key: str) -> Any:
        if hasattr(self.dataset, key):
            return getattr(self.dataset, key)
        else:
            raise AttributeError("Given dataset has no attribute \'{}\'".format(key))
