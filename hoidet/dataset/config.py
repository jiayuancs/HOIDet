"""
定义用于模型训练和验证的数据集
"""
from hoidet.dataset.base import DatasetInfo

__all__ = ['HICO_DET_INFO', 'VCOCO_INFO']

# TODO: 这里的数据集data_root参数采用了硬编码，后续可改进

# HICO_DET 数据集
HICO_DET_INFO = DatasetInfo(
    name="hicodet",
    data_root="/workspace/dataset/hicodet",
    partition_image={
        "train2015": "hico_20160224_det/images/train2015",
        "test2015": "hico_20160224_det/images/test2015"
    },
    partition_anno={
        "train2015": "instances_train2015.json",
        "test2015": "instances_test2015.json"
    },
    object_class_num=80,
    hoi_class_num=600,
    verb_class_num=117
)

# VCOCO 数据集
VCOCO_INFO = DatasetInfo(
    name="vcoco",
    data_root="/workspace/dataset/vcoco",
    partition_image={
        "train": "mscoco2014/train2014",
        "val": "mscoco2014/train2014",
        "trainval": "mscoco2014/train2014",
        "test": "mscoco2014/val2014"
    },
    partition_anno={
        "train": "instances_vcoco_train.json",
        "val": "instances_vcoco_val.json",
        "trainval": "instances_vcoco_trainval.json",
        "test": "instances_vcoco_test.json"
    },
    object_class_num=80,
    hoi_class_num=-1,  # vcoco 数据集不需要统计该类别信息
    verb_class_num=24
)

if __name__ == '__main__':
    print(HICO_DET_INFO)
    print(VCOCO_INFO)

    print(HICO_DET_INFO.get_anno_path("test2015"))
    print(HICO_DET_INFO.get_image_path("test2015"))

    print(VCOCO_INFO.get_anno_path("trainval"))
    print(VCOCO_INFO.get_image_path("trainval"))
