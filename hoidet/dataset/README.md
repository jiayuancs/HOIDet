# Dataset

目前仅支持 HICO-DET 和 V-COCO 两个数据集

## 获取数据集

按照如下仓库说明获取对应的数据集：

- 获取 HICO-DET 数据集： [fredzzhang/hicodet](https://github.com/fredzzhang/hicodet)
- 获取 V-COCO 数据集：[jiayuancs/vcoco](https://github.com/jiayuancs/vcoco)

完成上述操作后，这里假设 HICO-DET 和 V-COCO 数据集仓库在本地的路径分别为：

- HICO-DET: `/path/to/hicodet`
- V-COCO: `/path/to/vcoco`

将 [config.py](./config.py) 中 `data_root` 字段的值替换为上述路径。

## 加载数据集

```python
from hoidet.dataset import DataFactory
from hoidet.dataset import HICO_DET_INFO, VCOCO_INFO

# 加载 HICO-DET 的训练集
hicodet_trainset = DataFactory(
    partition=HICO_DET_INFO.get_training_partition(),
    dataset_info=HICO_DET_INFO
)

# 加载 HICO-DET 的测试集
hicodet_testset = DataFactory(
    partition=HICO_DET_INFO.get_testing_partition(),
    dataset_info=HICO_DET_INFO
)

# 加载 V-COCO 的训练集
vcoco_trainset = DataFactory(
    partition=VCOCO_INFO.get_training_partition(),
    dataset_info=VCOCO_INFO
)

# 加载 V-COCO 的测试集
vcoco_testset = DataFactory(
    partition=VCOCO_INFO.get_testing_partition(),
    dataset_info=VCOCO_INFO
)

# 将上述数据集实例传入 DataLoader 以训练或测试模型
```
