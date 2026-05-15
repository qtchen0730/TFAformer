"""
Function:
Transformer block for time series data.
date: 2025/06/11
Author: Qitong Chen and HuaYin
"""
import torch
from torch import nn
from models.TFAformer_attention_time_series import Attention_single_domain_tf
import torch.nn.functional as F



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class FeedForward(nn.Module):
    """    A simple feedforward neural network with two linear layers and a GELU activation function.
    Args:
        dim: 输入和输出的嵌入维度。
        hidden_dim: 隐藏层的维度。
        dropout: Dropout probability.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            # nn.LayerNorm(dim),  # --------》新添加LayerNorm层
        )

    def forward(self, x):
        return self.net(x)


class Transformer_block_single_domain_tf_mul_layer(nn.Module):
    """
    Transformer encoder block for time series data in a single source domain with time, time-frequency and frequency embeddings.
    ***************************可以实现多层encoder堆叠*********************************
    Args:
        dim: 输入和输出的嵌入维度。
        depth: Transformer encoder的层数。
        heads: 注意力头数量。
        dim_head: 每个注意力头的维度。
        mlp_dim: 前馈神经网络中隐藏层的维度。
        dropout: Dropout probability.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., args=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_single_domain_tf(dim, heads=heads, dim_head=dim_head, dropout=dropout, args=args),
                FeedForward(dim, mlp_dim, dropout=dropout)]))

    def forward(self, x_t, x_tf, x_f):
        """
        forward function
        Args:
            x_t:x_t_embedding 时间embedding数据
            x_tf:x_tf_embedding 时频embedding数据
            x_f:x_f_embedding 频率embedding数据
        Returns: Transformer编码器的输出。

        """
        for attn, ff in self.layers:
            # 对当前层的注意力模块进行调用，输出更新后的x
            x_t, x_tf, x_f = attn(x_t, x_tf, x_f)  # 添加标志位，先交叉注意力，后自注意力
            # 每个输入向量通过前馈网络 ff，并与原始值做残差连接（Residual）。类似标准 Transformer 的做法：输出 = 输入 + FeedForward(输入)
            x_t = ff(x_t) + x_t
            x_tf = ff(x_tf) + x_tf
            x_f = ff(x_f) + x_f
        return x_t, x_tf, x_f


if __name__ == '__main__':
    DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    Transformer_block_model = Transformer_block_single_domain_tf_mul_layer(dim=256, depth=2, heads=2, dim_head=128, mlp_dim=2*256, dropout=0.1).to(DEVICE)
    x_t = torch.randn(32, 128, 256).to(DEVICE)  # (batch_size, seq_len, dim)
    x_tf = torch.randn(32, 128, 256).to(DEVICE)  # (batch_size, seq_len, dim)
    x_f = torch.randn(32, 128, 256).to(DEVICE)  # (batch_size, seq_len, dim)
    x_t_attn_out, x_tf_attn_out, x_f_attn_out = Transformer_block_model(x_t, x_tf, x_f)
    print("x_t_attn_out shape:", x_t_attn_out.shape)
    print("x_tf_attn_out shape:", x_tf_attn_out.shape)
    print("x_f_attn_out shape:", x_f_attn_out.shape)
