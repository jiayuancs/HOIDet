# HOI metrics

评估 HOI detection 模型的性能

## HICO-DET

模型在测试集上的预测结果按如下格式使用 pickle 保存为 `.pkl` 文件：

```txt
[
  {
    'image_id': int,
    'boxes': [[x1,y1,x2,y2], ...],
    'h_idx': [int, ...],
    'o_idx': [int, ...],
    'hoi_score': [float, ...],
    'hoi_class': [int, ...]
  },
  ...
]
```

格式说明：

- `image_id` 是图片编号（即图片名称中的数字）
- `boxes` 是该图片中所有边界框的坐标，格式为`(x1,y1,x2,y2)`
- `h_idx[i]` 和 `o_idx[i]` 是 boxes 索引，构成该图片中的第 `i` 个 human-object pair
- `hoi_score[i]` 表示第 `i` 个 human-object pair 的置信度分数
- `hoi_class[i]` 表示第 `i` 个 human-object pair 的 HOI 类别编号

评测代码示例（假设模型预测结果保存在 `/path/to/pred.pkl`）：

```python
from hoidet.metrics import HICODetMetric

hicodet = HICODetMetric(pred_file_path="/path/to/pred.pkl")
hicodet.eval()  # 评测
print(f"mAP: {hicodet.get_full_map():.4f}\n"
      f"rare mAP: {hicodet.get_rare_map():.4f}\n"
      f"non-rare mAP: {hicodet.get_non_rare_map():.4f}")
```

## V-COCO

模型在测试集上的预测结果按如下格式使用 pickle 保存为 `.pkl` 文件（该格式与[s-gupta/v-coco](https://github.com/s-gupta/v-coco)相同）：

列表中每个元素描述了一个 human-object pair。

```txt
[
   {
   'image_id': int,
   'person_box': [x1, y1, x2, y2],
   '[action]_agent': float,
   '[action]_[role]': [x1, y1, x2, y2, s]
   },
   ...
]
```
格式说明：

- `image_id` 是图片编号（即图片名称中的数字）
- `person_box` 是该图片中的**一个**人框坐标
- `[action]_agent` 是当前这个 human-object pair 的置信度分数
- `[action]_[role]` 一个物体框和当前这个 human-object pair 的置信度分数
- `[action]` 是预测的动作类别名称（例如：`cut`, `read`, ...）
- `[role]` 是该动作的角色名称（要么是 `obj` 要么是 `instr`）

> 注意：在 HOI 检测任务中，`[action]_agent` 字段与 `[action]_[role]` 字段的最后一个元素(s)是相同的，
> 都是这对 human-object pair 的置信度分数

评测代码示例（假设模型预测结果保存在 `/path/to/pred.pkl`）：

```python
from hoidet.metrics import VCOCOMetric

vcoco = VCOCOMetric(pred_file_path="/path/to/pred.pkl")
vcoco.eval()  # 评测
print(f"role AP S1: {vcoco.get_map_s1():0.4f}")
print(f"role AP S2: {vcoco.get_map_s2():0.4f}")
```

## 自动评估

如果模型预测结果文件的命名格式满足正则表达式 `^(hicodet|vcoco)_\d\d\.pkl$`，则可以使用如下自动评估程序。

自动评估程序监控指定目录下的文件变动（递归），如果有新增的文件，
且文件名与正则表达式 `^(hicodet|vcoco)_\d\d\.pkl$` 匹配，则根据文件名前缀中的数据集名称，
自动调用相关数据集的评估工具，完成评估过程。

评估完成后，程序将评估结果保存在与模型预测结果文件相同的目录下，文件后缀为 `.txt`，文件名与模型预测结果文件相同。

```python
from hoidet.metrics import auto_eval

auto_eval(
    monitored_dir="/path/to/predict"
)
```
