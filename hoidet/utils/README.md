# utils

- [misc.py](./misc.py) 和 [box_ops.py](./box_ops.py) 来自 [DETR](https://github.com/facebookresearch/detr/)
- [transforms.py](./transforms.py) 提供各种数据变换方法
- [boxes.py](./boxes.py) 提供计算 IoU 的方法（TODO: 实际上可删除该文件）
- [relocate.py](./relocate.py) 用于将数据迁移到指定设备
- [association.py](./association.py) 用于判定预测结果中哪些是TP
- [meters.py](./meters.py) 存放数值指标，以及计算 AP
- [distributed.py](./distributed.py)
- [ops.py](./ops.py) 来自 [PVIC](https://github.com/fredzzhang/pvic/blob/main/ops.py)
