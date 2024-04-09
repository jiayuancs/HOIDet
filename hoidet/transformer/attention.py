"""
MultiHeadAttention: self-attention 和 cross-attention
"""

import math
import torch

from torch import nn, Tensor, Size
from typing import Optional

__all__ = ['MultiHeadAttention']


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 head_num: int,
                 dropout: float = 0.1,
                 value_dim: int = None) -> None:
        """
        Args:
            dim: query和key的维度
            head_num: head的个数
            dropout: dropout
            value_dim: value的维度，如果不指定，则默认与 dim 相同，即 query/key/value 具有相同的维度.
                       value_dim 也是注意力层最终输出的维度
        """
        super().__init__()

        self.dim = dim
        self.head_num = head_num
        self.head_kv_dim = dim // head_num
        # 每个head的维度拼接起来之后与输入/输出维度相同
        assert self.head_kv_dim * self.head_num == self.dim
        self.value_dim = value_dim if value_dim is not None else dim

        # 应用于注意力分数的dropout操作
        self.dropout_layer = nn.Dropout(dropout)
        # 多头注意力机制输出层的线性变换
        self.output_projection_layer = nn.Linear(self.value_dim, self.value_dim)
        self.softmax = nn.Softmax(dim=1)

        # 用于缩放点积注意力分数的计算
        self.scale = 1.0 / math.sqrt(self.head_kv_dim)

        # 存储注意力分数，以便进行可视化工作
        self.attn = None

    @staticmethod
    def _dot_product(query: Tensor, key: Tensor):
        """
        计算query和key的相似度(点积注意力分数)

        Args:
            query: 形状为 [seq_len_q, batch_size, head_num, head_dim]
            key: 形状为 [seq_len_k, batch_size, head_num, head_dim]

        Returns: 形状为 [seq_len_q, seq_len_k, batch_size, head_num]

        Note: batch_size为何放在第二维度

            因为batch first意味着模型的输入在内存中存储时，先存储第一个sequence，再存储第二个......。
            而seq_len first意味着不同序列中同一个时刻对应的输入单元在内存中是毗邻的，这样进行batch运算时，
            可以有效地利用memory的局部性
        """
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    @staticmethod
    def _prepare_key_value_mask(mask: Tensor, query_shape: Size, key_shape: Size):
        """
        Args:
            mask: 形状为 [seq_len_q, seq_len_k, batch_size]，其中seq_len_q和batch_size可为1，
                      此时会使用广播
            query_shape: 即 [seq_len_q, batch_size, head_num, head_dim]
            key_shape: 即 [seq_len_k, batch_size, head_num, head_dim]

        Returns: 形状为 [seq_len_q, seq_len_k, batch_size, 1]
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)
        return mask

    @staticmethod
    def _prepare_key_padding_mask(mask: Tensor, query_shape: Size, key_shape: Size):
        """
        Args:
            mask: 形状为 [seq_len_k, batch_size] 的 bool 张量,
                mask[i,b]表示对于批量中的第b个样本中的第i个key向量是否有效（即是否为非padding），
                True 表示有效，False 表示该分量是 padding
            query_shape: 即 [seq_len_q, batch_size, head_num, head_dim]
            key_shape: 即 [seq_len_k, batch_size, head_num, head_dim]

        Returns: 形状为 [seq_len_q, seq_len_k, batch_size, 1]

        """
        q_len, k_len = query_shape[0], key_shape[0]
        batch_size = query_shape[1]
        mask = mask.unsqueeze(1).expand(batch_size, q_len, k_len)
        return mask

    def forward(self, *,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_value_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None):
        """
        该 MultiHeadAttention 不会对 query、key、value 做线性变换，
        而是直接计算 query 和 key 的相似度，然后对 value 进行求和

        Args:
            query: 形状为 [seq_len_q, batch_size, dim]
            key: 形状为 [seq_len, batch_size, dim]
            value: 形状为 [seq_len, batch_size, dim]
            key_value_mask: 形状为 [seq_len_q, seq_len_k, batch_size]，
               key_value_mask[i,j,b]表示对于批量中的第b个样本中的第i个查询向量，其是否能够访问到第j个键值对，
               True 表示能访问，False 表示不能访问
            key_padding_mask: 形状为 [seq_len_k, batch_size] 的 bool 张量,
                key_padding_mask[i,b]表示对于批量中的第b个样本中的第i个key向量，其第i个分量是否有效（即是否为非padding），
                True 表示有效，False 表示该分量是 padding

        Returns:

        Note: 参数'*'的作用

            这里使用参数'*'用于指定后续参数必须以关键字参数的形式传递，
            而不能使用位置参数。这种方式可以增加代码的可读性和可维护性，
            因为它明确指定了每个参数的用途，而不会引发歧义。

        """

        seq_len, batch_size, _ = query.shape
        if key_value_mask is not None:
            key_value_mask = self._prepare_key_value_mask(key_value_mask, query.shape, key.shape)
        if key_padding_mask is not None:
            key_padding_mask = self._prepare_key_padding_mask(key_padding_mask, query.shape, key.shape)

        # 变换形状，以应用于多头注意力机制，形状为 [seq_len, batch_size, head_num, head_dim]
        query = query.view(*query.shape[:-1], self.head_num, self.head_kv_dim)
        key = key.view(*key.shape[:-1], self.head_num, self.head_kv_dim)
        value = value.view(*value.shape[:-1], self.head_num, self.value_dim // self.head_num)

        # 计算注意力分数(缩放点积注意力评分函数)
        # [seq_len_q, seq_len_k, batch_size, head_num]
        scores = self._dot_product(query, key) * self.scale

        if key_value_mask is not None:
            # -inf经过softmax后就变成了0
            scores.masked_fill_(key_value_mask.eq(0), float('-inf'))
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask.eq(0), float('-inf'))

        # 对value进行加权求和
        attn = self.softmax(scores)
        attn = self.dropout_layer(attn)
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)

        # 保存注意力分数，以便用于后续可视化工作
        self.attn = attn.detach()

        # 将多个头的输出拼接到一起
        x = x.reshape(seq_len, batch_size, -1)

        # 返回注意力运算结果和注意力分数
        return self.output_projection_layer(x), self.attn


if __name__ == '__main__':
    torch.manual_seed(42)
    self_attn = MultiHeadAttention(
        dim=10,
        head_num=2
    )

    x = torch.randn((4, 1, 10))
    y, weight = self_attn(
        query=x,
        key=x,
        value=x
    )

    print(y.shape)
    print(weight.shape)
