import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from hoidet.dataset import HICODet

__all__ = ['plot_dual_axis']


def plot_dual_axis(x, x_label,
                   y1, y1_label,
                   y2, y2_label,
                   title="", y1_color='#1f77b4', y2_color='#ff7f0e', figsize=None):
    """
    在同一画布上绘制两个折线，分别使用两个不同刻度的纵轴
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # 绘制第一条线并设置第一个坐标轴
    ax1.plot(x, y1, label=y1_label, color=y1_color)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.tick_params('y', colors=y1_color)

    # 创建第二个坐标轴，并绘制第二条线
    ax2 = ax1.twinx()
    ax2.plot(x, y2, label=y2_label, color=y2_color)
    ax2.set_ylabel(y2_label, color=y2_color)
    ax2.tick_params('y', colors=y2_color)

    # 添加标题和图例
    plt.title(title)
    fig.tight_layout()


class HICODetAPChart:
    def __init__(self, ap, dataset: HICODet):
        """
        Args:
            ap: 每个HOI类别的AP值
            dataset: 数据集对象
        """
        ap = np.asarray(ap)
        hoi_instance_num = np.asarray(dataset.hoi_instance_num)
        hoi_class_names = np.asarray(dataset.hoi_class_names)

        self.verb_names = dataset.verbs
        self.object_names = dataset.objects

        # 按AP从小到大排序
        sorted_indices = np.argsort(ap)
        self.ap = ap[sorted_indices]
        self.hoi_instance_num = hoi_instance_num[sorted_indices]
        self.hoi_class_names = hoi_class_names[sorted_indices]
        self.x = np.arange(len(self.ap))

    def plot_ap(self, title="AP on HICO-DET"):
        """绘制AP曲线"""
        plt.plot(self.x, self.ap)
        plt.xlabel("HOI class (sort by ap)")
        plt.ylabel("AP")
        plt.title(title)
        return self.hoi_class_names

    def plot_ap_and_instance_num(self, title="AP and Instance num on HICO-DET", figsize=None):
        """同时绘制AP曲线和每个HOI类别实例个数的曲线"""
        plot_dual_axis(self.x, "HOI class (sort by ap)",
                       self.ap, "AP",
                       self.hoi_instance_num, "HOI instance num",
                       title=title, figsize=figsize)
        return self.hoi_class_names

    def plot_ap_per_verb(self, title="mAP per verb"):
        """统计每个verb对应的AP，绘制曲线"""
        verb_ap = {k: [] for k in self.verb_names}
        for hoi_name, ap in zip(self.hoi_class_names, self.ap):
            verb_name = hoi_name.split(" ")[0]
            assert verb_name in verb_ap.keys(), "未知的动作类别"
            verb_ap[verb_name].append(ap)

        for verb_name in self.verb_names:
            ap_per_verb = np.asarray(verb_ap[verb_name]).mean()
            verb_ap[verb_name] = ap_per_verb

        # 按ap从小到大排序
        verb_ap_sorted = sorted(verb_ap.items(), key=lambda x: x[1])
        verb_names = []  # verb_name[i]表示第i个动作的名称
        aps = []  # aps[i]表示第i个动作的mAP
        for key, value in verb_ap_sorted:
            verb_names.append(key)
            aps.append(value)

        # 绘图
        plt.plot(range(len(verb_names)), aps)
        plt.xlabel("Verb class (sort by mAP)")
        plt.ylabel("mAP")
        plt.title(title)
        return verb_names

    def plot_ap_by_verb(self, verb_name: str, title="AP of verb"):
        """绘制指定动作的AP曲线"""
        object_ap = {}
        for hoi_name, ap in zip(self.hoi_class_names, self.ap):
            verb_n, object_n = hoi_name.split(" ")
            assert verb_n in self.verb_names, "未知的动作类别"
            assert object_n in self.object_names, "未知的物体类别"

            if verb_name == verb_n:
                object_ap[object_n] = ap

        # 按ap从小到大排序
        object_ap_sorted = sorted(object_ap.items(), key=lambda x: x[1])
        object_names = []
        aps = []
        for key, value in object_ap_sorted:
            object_names.append(key)
            aps.append(value)
        dot_cnt = len(object_names)

        # # 自适应宽高
        w, h = 8., 8.
        w = max(w, dot_cnt * 0.5)

        # 绘图
        plt.figure(figsize=(w, h))
        plt.xticks(rotation=40)
        plt.scatter(object_names, aps, marker='o', alpha=0.5, s=80)
        plt.plot(object_names, aps)
        plt.xlabel("Object class name (sort by mAP)")
        plt.ylabel("AP")
        plt.title(title)
        plt.tight_layout()
        return object_names

    def plot_all(self, save_dir="./img"):
        """绘制所有支持的图，保存在指定目录下"""
        os.makedirs(save_dir, exist_ok=True)

        self.clean()
        self.plot_ap()
        self.save(os.path.join(save_dir, "ap.png"))

        self.clean()
        self.plot_ap_and_instance_num()
        self.save(os.path.join(save_dir, "ap_and_instance_num.png"))

        self.clean()
        verb_names = self.plot_ap_per_verb()
        self.save(os.path.join(save_dir, "ap_per_verb.png"))

        for idx, name in tqdm(enumerate(verb_names)):
            self.clean()
            self.plot_ap_by_verb(verb_name=name, title=f"AP of {name} (#{idx:03})")
            self.save(os.path.join(save_dir, f"ap_of_verb_{idx:03}_{name}.png"))

    @staticmethod
    def save(save_path):
        plt.savefig(save_path)

    @staticmethod
    def clean():
        plt.close()
        plt.clf()


if __name__ == '__main__':
    import pickle
    from hoidet.dataset import HICO_DET_INFO

    dataset = HICODet(
        dataset_info=HICO_DET_INFO,
        partition="test2015"
    )

    with open("../../data/pred_ap_pvic.pkl", "rb") as f:
        ap = pickle.load(f)

    chart = HICODetAPChart(ap, dataset)
    chart.plot_all(save_dir="../../data/img")
