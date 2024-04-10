"""
Transformer Decoder

参考 Conditional DETR，cross-attention 中的 query/key 采用与
位置编码拼接的方案

在下面的代码中：
    - pos 表示位置编码
    - sa 表示 self-attention
    - ca 表示 cross-attention
"""
import copy
import torch
from torch import nn, Tensor
from typing import Optional

from hoidet.transformer.attention import MultiHeadAttention

__all__ = ['TransformerDecoderLayer', 'TransformerDecoder']


class TransformerDecoderLayer(nn.Module):
    def __init__(self, query_dim: int, cross_kv_dim: int, head_num: int, ffn_interm_dim: int = None,
                 dropout: float = 0.1,
                 sa_pos_dim: int = None, ca_query_pos_dim: int = None, ca_key_pos_dim: int = None):
        """
        参数中，sa 表示 self-attention, ca 表示 cross-attention

        Args:
            query_dim: query 向量的维度
            cross_kv_dim: cross-attention 中 key/value 向量的维度
            head_num: head的个数
            ffn_interm_dim: Decoder中的全连接层中隐层特征维度，默认为 4*query_dim
            dropout: dropout
            sa_pos_dim: self-atttention 中的位置编码维度，如不指定，则默认与 query_dim 相同
            ca_query_pos_dim: cross-attention 中 query 向量的位置编码维度，如不指定，则默认与 cross_kv_dim 相同
            ca_key_pos_dim: cross-attention 中 key 向量的位置编码维度，如不指定，则默认与 cross_kv_dim 相同
        """
        super().__init__()

        self.query_dim = query_dim
        self.cross_kv_dim = cross_kv_dim
        self.head_num = head_num
        self.ffn_interm_dim = ffn_interm_dim if ffn_interm_dim is not None else self.query_dim * 4
        self.sa_pos_dim = sa_pos_dim if sa_pos_dim is not None else query_dim
        self.ca_query_pos_dim = ca_query_pos_dim if ca_query_pos_dim is not None else cross_kv_dim
        self.ca_key_pos_dim = ca_key_pos_dim if ca_key_pos_dim is not None else cross_kv_dim
        # 确保维度可被每个head均分
        assert self.query_dim % self.head_num == 0

        # 在 decoder 的 self-attention 阶段，query/key 采用与位置编码相加的方案，
        # 因此 query/key/value 的维度都是 query_dim
        self.self_attention = MultiHeadAttention(
            dim=query_dim,
            head_num=head_num,
            dropout=dropout
        )
        # self-attention(sa) 阶段相关的线性变换
        self.sa_q_linear = nn.Linear(query_dim, query_dim)
        self.sa_k_linear = nn.Linear(query_dim, query_dim)
        self.sa_v_linear = nn.Linear(query_dim, query_dim)
        # 位置编码相关的变换
        self.sa_q_pos_linear = nn.Linear(self.sa_pos_dim, query_dim)
        self.sa_k_pos_linear = nn.Linear(self.sa_pos_dim, query_dim)

        # 交叉注意力的 query/key 采用与位置编码拼接的方案，因此其维度是 2*query_dim
        # value 的维度仍为 dim
        self.cross_attention = MultiHeadAttention(
            dim=query_dim * 2,
            head_num=head_num,
            dropout=dropout,
            value_dim=query_dim  # 指定输出维度为 query_dim
        )
        # cross-attenton(ca) 阶段相关的线性变换
        self.ca_q_linear = nn.Linear(query_dim, query_dim)
        self.ca_k_linear = nn.Linear(cross_kv_dim, query_dim)
        self.ca_v_linear = nn.Linear(cross_kv_dim, query_dim)
        # 位置编码相关的变换
        self.ca_q_pos_linear = nn.Linear(self.ca_query_pos_dim, query_dim)
        self.ca_k_pos_linear = nn.Linear(self.ca_key_pos_dim, query_dim)

        # LayerNorm 中包含了可学习的参数，故这里实例化了3个
        self.layer_norm1 = nn.LayerNorm(query_dim)
        self.layer_norm2 = nn.LayerNorm(query_dim)
        self.layer_norm3 = nn.LayerNorm(query_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, self.ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.ffn_interm_dim, query_dim)
        )

    def forward(self,
                query: Tensor,
                key_value: Tensor,
                sa_pos: Tensor,
                ca_query_pos: Tensor,
                ca_key_pos: Tensor,
                sa_key_value_mask: Optional[Tensor] = None,
                sa_key_padding_mask: Optional[Tensor] = None,
                ca_key_value_mask: Optional[Tensor] = None,
                ca_key_padding_mask: Optional[Tensor] = None):
        """
        参数中，sa 表示 self-attention, ca 表示 cross-attention

        Args:
            query: 形状为 [seq_len_q, batch_size, query_dim] 的 query 向量
            key_value: 形状为 [seq_len_kv, batch_size, cross_kv_dim], 是 cross-attention 的 key/value 向量
            sa_pos: 形状为 [seq_len_q, batch_size, sa_pos_dim], self-attention 的位置编码
            ca_query_pos: 形状为 [seq_len_kv, batch_size, ca_query_pos_dim], cross-attention 的 query 的位置编码
            ca_key_pos: 形状为 [seq_len_kv, batch_size, ca_key_pos_dim], cross-attention 的 key 的位置编码
            sa_key_value_mask: 形状为 [seq_len_q, seq_len_k, batch_size]，用于 self-attention 模块，
                sa_key_value_mask[i,j,b]表示对于批量中的第b个样本中的第i个查询向量，其是否能够访问到第j个键值对，
                True 表示能访问，False 表示不能访问
            sa_key_padding_mask: 形状为 [seq_len_k, batch_size] 的 bool 张量,用于 self-attention 模块，
                sa_key_padding_mask[i,b]表示对于批量中的第b个样本中的第i个key向量，其第i个分量是否有效（即是否为非padding），
                True 表示有效，False 表示该分量是 padding
            ca_key_value_mask: 含义同 sa_key_value_mask，只不过是针对 cross-attention 模块的
            ca_key_padding_mask: 含义同 sa_key_padding_mask，只不过是针对 cross-attention 模块的

        Returns:

        """

        # -----------------self-attention 阶段-----------------
        # 位置编码与 query 和 key 直接相加
        q = self.sa_q_linear(query) + self.sa_q_pos_linear(sa_pos)
        k = self.sa_k_linear(query) + self.sa_k_pos_linear(sa_pos)
        v = self.sa_v_linear(query)
        sa_result, _ = self.self_attention(
            query=q, key=k, value=v,
            key_value_mask=sa_key_value_mask,
            key_padding_mask=sa_key_padding_mask
        )

        # -----------------add & norm-----------------
        query = self.layer_norm1(query + self.dropout_layer(sa_result))

        # -----------------cross-attention 阶段-----------------
        q = self.ca_q_linear(query)  # [seq_len_q, batch_size, head_dim * head_num]
        k = self.ca_k_linear(key_value)
        v = self.ca_v_linear(key_value)
        q_pos = self.ca_q_pos_linear(ca_query_pos)
        k_pos = self.ca_k_pos_linear(ca_key_pos)

        # 将 q 变形为 [seq_len_q, batch_size, head_num, head_dim]，以便拼接位置编码
        seq_len_q, batch_size, query_dim = q.shape
        q = q.view(seq_len_q, batch_size, self.head_num, query_dim // self.head_num)
        q_pos = q_pos.view(seq_len_q, batch_size, self.head_num, query_dim // self.head_num)
        q = torch.cat([q, q_pos], dim=3).view(seq_len_q, batch_size, query_dim * 2)

        # 同理，对 k 进行变形，然后拼接位置编码
        seq_len_k, _, _ = k.shape
        k = k.view(seq_len_k, batch_size, self.head_num, query_dim // self.head_num)
        k_pos = k_pos.view(seq_len_k, batch_size, self.head_num, query_dim // self.head_num)
        k = torch.cat([k, k_pos], dim=3).view(seq_len_k, batch_size, query_dim * 2)

        # cross-attention
        ca_result, _ = self.cross_attention(
            query=q, key=k, value=v,
            key_value_mask=ca_key_value_mask,
            key_padding_mask=ca_key_padding_mask
        )

        # -----------------add & norm-----------------
        query = self.layer_norm2(query + self.dropout_layer(ca_result))

        # -----------------Feed Forward, Add & Norm-----------------
        query = self.layer_norm3(query + self.dropout_layer(self.ffn(query)))

        return query


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer,
                 layer_num: int, return_intermediate=True):
        """
        Args:
            decoder_layer: 实例化的 TransformerDecoderLayer
            layer_num: 层数
            return_intermediate: 是否返回每一层的输出结果
        """
        super().__init__()
        assert layer_num > 0
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(layer_num)])
        self.layer_num = layer_num
        self.layer_norm = nn.LayerNorm(decoder_layer.query_dim)
        self.return_intermediate = return_intermediate

        # 重新初始化每一层的参数
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                query: Tensor,
                key_value: Tensor,
                sa_pos: Tensor,
                ca_query_pos: Tensor,
                ca_key_pos: Tensor,
                sa_key_value_mask: Optional[Tensor] = None,
                sa_key_padding_mask: Optional[Tensor] = None,
                ca_key_value_mask: Optional[Tensor] = None,
                ca_key_padding_mask: Optional[Tensor] = None):
        """
        参数中，sa 表示 self-attention, ca 表示 cross-attention

        Args:
            query: 形状为 [seq_len_q, batch_size, query_dim] 的 query 向量
            key_value: 形状为 [seq_len_kv, batch_size, cross_kv_dim], 是 cross-attention 的 key/value 向量
            sa_pos: 形状为 [seq_len_q, batch_size, sa_pos_dim], self-attention 的位置编码
            ca_query_pos: 形状为 [seq_len_kv, batch_size, ca_query_pos_dim], cross-attention 的 query 的位置编码
            ca_key_pos: 形状为 [seq_len_kv, batch_size, ca_key_pos_dim], cross-attention 的 key 的位置编码
            sa_key_value_mask: 形状为 [seq_len_q, seq_len_k, batch_size]，用于 self-attention 模块，
                sa_key_value_mask[i,j,b]表示对于批量中的第b个样本中的第i个查询向量，其是否能够访问到第j个键值对，
                True 表示能访问，False 表示不能访问
            sa_key_padding_mask: 形状为 [seq_len_k, batch_size] 的 bool 张量,用于 self-attention 模块，
                sa_key_padding_mask[i,b]表示对于批量中的第b个样本中的第i个key向量，其第i个分量是否有效（即是否为非padding），
                True 表示有效，False 表示该分量是 padding
            ca_key_value_mask: 含义同 sa_key_value_mask，只不过是针对 cross-attention 模块的
            ca_key_padding_mask: 含义同 sa_key_padding_mask，只不过是针对 cross-attention 模块的

        Returns:
            如果 self.return_intermediate=True, 则返回形状为 [layer_num, seq_len_q, batch_size, query_dim] 的向量，
            表示每一层的计算结果；
            如果 self.return_intermediate=False, 则返回形状为 [1, seq_len_q, batch_size, query_dim] 的向量，
            表示最后一层的计算结果；
            因此，无论哪种情况，都可以用 output[-1] 获取最后一层的计算结果

        """
        output = query
        intermediate = []
        for layer in self.layers:
            output = layer(
                query=output,
                key_value=key_value,
                sa_pos=sa_pos,
                ca_query_pos=ca_query_pos,
                ca_key_pos=ca_key_pos,
                sa_key_value_mask=sa_key_value_mask,
                sa_key_padding_mask=sa_key_padding_mask,
                ca_key_value_mask=ca_key_value_mask,
                ca_key_padding_mask=ca_key_padding_mask
            )
            if self.return_intermediate:
                intermediate.append(self.layer_norm(output))

        if self.return_intermediate:
            output = torch.stack(intermediate)
        else:
            output = self.layer_norm(output).unsqueeze(0)
        return output


if __name__ == '__main__':
    batch_size = 4
    query_len = 2
    query_dim = 10

    query = torch.randn((query_len, batch_size, query_dim))
    sa_pos = torch.randn((query_len, batch_size, query_dim))

    key_value = torch.randn((query_len, batch_size, query_dim))
    ca_q_pos = torch.randn((query_len, batch_size, query_dim))
    ca_k_pos = torch.randn((query_len, batch_size, query_dim))

    decoder_layer = TransformerDecoderLayer(
        query_dim=query_dim,
        cross_kv_dim=key_value.shape[-1],
        head_num=2,
        ffn_interm_dim=None
    )
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        layer_num=6,
        return_intermediate=False
    )

    result = decoder(
        query=query,
        key_value=key_value,
        sa_pos=sa_pos,
        ca_query_pos=ca_q_pos,
        ca_key_pos=ca_k_pos
    )

    print(result.shape)
