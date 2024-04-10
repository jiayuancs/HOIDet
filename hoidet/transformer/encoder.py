"""
Transformer Encoder

位置编码与 query/key 直接相加

"""

from torch import nn, Tensor
from typing import Optional

from hoidet.transformer.attention import MultiHeadAttention

__all__ = ['TransformerEncoderLayer', 'TransformerEncoder']


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, head_num: int, ffn_interm_dim: int = None, dropout: float = 0.1,
                 pos_dim: int = None):
        """
        Args:
            dim: 输入和输出特征的维度
            head_num: head的数量
            ffn_interm_dim: Encoder末端的全连接层中隐层特征维度
            dropout: dropout
            pos_dim: 输入的位置编码的维度，如果为 None，则表示与 dim 相同
        """
        super().__init__()

        self.dim = dim
        self.head_num = head_num
        self.ffn_interm_dim = ffn_interm_dim if ffn_interm_dim is not None else self.dim * 4
        self.pos_dim = pos_dim if pos_dim is not None else dim

        # 自注意力模块，该模块中没有对 query/key/value 做线性变换
        self.self_attention = MultiHeadAttention(
            dim=dim,
            head_num=head_num,
            dropout=dropout
        )
        # 对 query/key/value 做线性变换
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        # 对位置编码做线性变换
        self.q_pos_linear = nn.Linear(self.pos_dim, dim)
        self.k_pos_linear = nn.Linear(self.pos_dim, dim)

        # LayerNorm 中包含了可学习的参数，故这里实例化了两个
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.ffn_interm_dim, dim)
        )

    def forward(self, x: Tensor, pos: Tensor,
                key_value_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None):
        """
        Args:
            x: 形状为 [seq_len, batch_size, dim]
            pos: 形状为 [pos_dim, batch_size, dim]
            key_value_mask: 形状为 [seq_len_q, seq_len_k, batch_size]，
               key_value_mask[i,j,b]表示对于批量中的第b个样本中的第i个查询向量，其是否能够访问到第j个键值对，
               True 表示能访问，False 表示不能访问
            key_padding_mask: 形状为 [seq_len_k, batch_size] 的 bool 张量,
                key_padding_mask[i,b]表示对于批量中的第b个样本中的第i个key向量，其第i个分量是否有效（即是否为非padding），
                True 表示有效，False 表示该分量是 padding

        Returns:

        """
        # TODO: 尝试在self-attention之前进行层归一化
        # x = self.layer_norm(x)

        # 位置编码与 query 和 key 直接相加
        q = self.q_linear(x) + self.q_pos_linear(pos)
        k = self.k_linear(x) + self.k_pos_linear(pos)
        v = self.v_linear(x)

        # Multi-Head self-attention
        result, attn_score = self.self_attention(
            query=q, key=k, value=v,
            key_value_mask=key_value_mask,
            key_padding_mask=key_padding_mask
        )

        # Add & Norm
        x = self.layer_norm1(x + self.dropout_layer(result))

        # Feed Forward, Add & Norm
        x = self.layer_norm2(x + self.dropout_layer(self.ffn(x)))

        # 返回encoder层计算结果和注意力分数
        return x, attn_score


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int = 256, head_num: int = 8, layer_num: int = 2, dropout: float = 0.1):
        """
        Args:
            dim: 输入和输出特征的维度
            head_num: head的数量
            layer_num: encoder的层数
            dropout: dropout
        """
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            dim=dim, head_num=head_num,
            ffn_interm_dim=dim * 4, dropout=dropout
        ) for _ in range(layer_num)])

    def forward(self, x: Tensor, pos: Tensor):
        """
        Args:
            x: 形状为 [seq_len, batch_size, dim]
            pos: 形状为 [pos_dim, batch_size, dim]

        Returns:

        """
        attn_weights = []
        for layer in self.layers:
            x, attn_score = layer(x, pos)
            attn_weights.append(attn_score)
        return x, attn_weights


if __name__ == '__main__':
    import torch

    torch.manual_seed(42)

    encoder = TransformerEncoder(
        dim=10,
        head_num=2,
        layer_num=6
    )

    x = torch.randn((4, 2, 10))
    pos = torch.randn((4, 2, 10))
    y, attn_score = encoder(x, pos)

    print(y.shape)
