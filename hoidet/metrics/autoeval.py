"""
监控指定目录下的新增文件，自动评估模型性能
"""

import os
import re
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from hoidet.metrics import HICODetMetric, VCOCOMetric

# 数据集名称与其对应的评价工具
EVAL_MAP = {
    "hicodet": HICODetMetric,
    "vcoco": VCOCOMetric
}


def eval(dirname, filename, suffix="txt"):
    """评估指定的预测文件"""
    filepath = os.path.join(dirname, filename)
    prefix, suffix = filename.split(".")
    pred_label = os.path.join(os.path.basename(dirname), filename)
    eval_file = os.path.join(dirname, f"{prefix}.txt")

    # 如果该文件已经被评测过，则直接跳过
    if os.path.exists(eval_file):
        print(f"SKIP: {pred_label}, 评测结果文件已存在")
        return

    # 文件名必须是数据集名称
    dataset_name = prefix.split("_")[0]
    if dataset_name not in EVAL_MAP.keys():
        print(f"SKIP: {pred_label}, 不支持{dataset_name}数据集")
        return

    # 输出评测结果
    print(f"{pred_label}: evaluating")
    metric = EVAL_MAP[dataset_name](pred_file_path=filepath)
    metric.eval()
    result = f"{pred_label}: {metric.summary()}"
    print(result)

    # 将评测结果保存到文件
    with open(eval_file, "w", encoding="utf-8") as fd:
        fd.write(result + "\n")


class EvalHandler(PatternMatchingEventHandler):
    def __init__(self, filename_pattern):
        super().__init__()
        self.filename_reg = filename_pattern

    def on_closed(self, event):
        """监控文件关闭事件"""
        # 跳过目录
        if event.is_directory:
            return

        dir_name = os.path.dirname(event.src_path)
        file_name = os.path.basename(event.src_path)
        if re.match(self.filename_reg, file_name):
            # 如果文件名与正则表达式匹配，则执行评估过程
            eval(
                dirname=dir_name,
                filename=file_name,
                suffix="txt"
            )


if __name__ == "__main__":
    # 要监控的目录
    monitored_dir = "../../predict"

    # 要匹配的文件名称正则表达式
    file_pattern = r"^(hicodet|vcoco)_\d\d\.pkl$"

    # 创建观察者对象并指定要监控的目录
    event_handler = EvalHandler(file_pattern)
    observer = Observer()
    observer.schedule(event_handler, monitored_dir, recursive=True)
    observer.start()

    try:
        print(f"Monitoring directory: {monitored_dir}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
