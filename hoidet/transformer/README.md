# Transformer

提供基础的 Transformer 组件

## MultiHeadAttention

**注意**：MultiHeadAttention 并**没有**对 query、key、value 作线性变换。因此如果需要单独使用该模块，需要在执行注意力层的 forward 
之前，对 query、key、value 作线性变换。

**实例化参数**：

- `dim` 是 query 和 key 的维度
- `head_num` 是 head 的个数
- `dropout` 是 dropout 层的参数，默认为 0.1
- `value_dim` 是 value 的维度，也是注意力层最终**输出的特征维度**，默认为 None，表示与 `dim` 相同；

**forward 参数**：

- `query` 形状为 `[seq_len_q, batch_size, dim]`
- `key` 形状为 `[seq_len, batch_size, dim]`
- `value` 形状为 `[seq_len, batch_size, dim]`
- `key_value_mask` 形状为 `[seq_len_q, seq_len_k, batch_size]`. `key_value_mask[i,j,b]` 表示对于批量中的第 b 个样本中的第 i 个 query， 
其是否能够访问到第 j 个 key-value，**True 表示能访问**，False 表示不能访问.
- `key_padding_mask` 形状为 `[seq_len_k, batch_size]`. `key_padding_mask[i,b]`表示批量中的第 b 个样本中的第 i 个 key 向量是否有效，
即是否为非 padding. **True 表示有效**，False 表示该 key 是 padding.

**forward 返回值**：

- 返回注意力运算结果和注意力分数

**使用示例**：

```Python
from hoidet.transformer import MultiHeadAttention

# 实例化一个多头注意力层
attention_layer = MultiHeadAttention(
    dim=256,        # query和key的维度
    head_num=8,     # head的个数
    dropout=0.1,    # dropout
    value_dim=None  # value的维度与 dim 相同
)

# 进行线性变换，得到 q/k/v
# q, k, v = ...

# 执行注意力过程
embed, score = attention_layer(
    query=q, key=k, value=v
)
```


