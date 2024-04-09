"""
生成位置编码
"""
import math
import numpy as np
import torch

__all__ = ['sin_cos_position']


def sin_cos_position(dim: int, temperature: float = 20, max_len: int = 5000):
    """
    获取位置编码矩阵
    :param dim: 位置编码向量的维度，与词向量的维度相同
    :param temperature: 正余弦编码的T参数，根据 DAB-DETR，这里将其默认设置为 20
    :param max_len: 位置的最大个数，即句子的最大长度
    :return: 形状为 [max_len, 1, dim]
    """
    encodings = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

    two_i = torch.arange(0, dim, 2, dtype=torch.float32)
    div_term = torch.exp(- two_i * (math.log(temperature) / dim))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)

    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


def _test_positional_encoding(save_path="./tmp.png"):
    """可视化位置编码"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    pe = sin_cos_position(dim=20, max_len=100)
    plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional encoding")
    plt.savefig(save_path)


if __name__ == '__main__':
    _test_positional_encoding()
