"""
读取评测结果，进行可视化
"""
import os
import argparse
from typing import List
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default="./predict/pred_61", type=str,
                    help="评测结果所在的目录")
args = parser.parse_args()
# print(args)


def get_newest_dir(directory, prefix="pred_"):
    max_number = -1
    max_subdir = None

    # 遍历指定目录下的所有子目录
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith(prefix):
            try:
                # 解析子目录的编号
                number = int(subdir[len(prefix):])
                if number > max_number:
                    max_number = number
                    max_subdir = subdir_path
            except ValueError:
                print(f"WARNING: Skip {subdir_path}")

    return max_subdir


def get_map_from_log(log_str_list: List[str]) -> dict:
    """
    Args:
        log_str_list(List[str]): 评测结果日志列表(按照文件名排序)，示例如下：
            ['pred_61/hicodet_01.pkl: mAP: 0.2554, rare mAP: 0.2011, non-rare mAP: 0.2716', ...]

    Returns:
    """
    all_results = {"index": []}
    for log in log_str_list:
        filename, results = log.split(":", 1)  # 按第一个冒号分割
        idx = int(filename.split("_")[-1].split(".")[0])
        all_results["index"].append(idx)
        for res in results.split(","):
            key, value = res.split(":")
            key = key.strip()
            value = float(value.strip())
            if key in all_results.keys():
                all_results[key].append(value)
            else:
                all_results[key] = [value]
    return all_results


def draw(title, results_dict, save_path):
    x = results_dict["index"]
    for i, (key, value) in enumerate(results_dict.items()):
        if key == "index":
            continue
        plt.plot(x, value, label=key)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("mAP")
    plt.title(title)
    plt.savefig(save_path)


if __name__ == "__main__":
    dir_root = args.dir

    # 获取所有评测结果
    log_list = []
    for filename in os.listdir(dir_root):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(dir_root, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            log = f.read().strip()
            log_list.append(log)
    log_list = sorted(log_list)

    # plt.figure(figsize=(5, 5))
    draw(title=os.path.basename(args.dir),
         results_dict=get_map_from_log(log_list),
         save_path=os.path.join(args.dir, "mAP.jpg"))
