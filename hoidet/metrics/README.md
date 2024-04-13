# HOI metrics

测试 HOI detection 模型的性能

## HICO-DET

模型在测试集上的预测结果按如下格式保存为 json 文件：

```json
[
  {
    'image_id': int,
    'boxes': [
      [
        x1,
        y1,
        x2,
        y2
      ],
      ...
    ],
    'h_idx': [
      int,
      ...
    ],
    'o_idx': [
      int,
      ...
    ],
    'hoi_score': [
      float,
      ...
    ],
    'hoi_class': [
      int,
      ...
    ]
  }
]
```

格式说明：

- `image_id` 是图片编号（即图片名称中的数字）
- `boxes` 是该目标中所有边界框的坐标，格式为`(x1,y1,x2,y2)`
- `h_idx[i]` 和 `o_idx[i]` 是 boxes 索引，构成该图片中的第 `i` 个 human-object pair
- `hoi_score[i]` 表示第 `i` 个 human-object pair 的置信度分数
- `hoi_class[i]` 表示第 `i` 个 human-object pair 的 HOI 类别编号

评测代码示例（假设模型预测结果保存在 `/path/to/pred.json`）：

```python
from hoidet.metrics import HICODetMetric

hicodet = HICODetMetric(pred_file_path="/path/to/pred.json")
print(f"mAP: {hicodet.get_full_map():0.4f}")
```

## V-COCO

TODO

