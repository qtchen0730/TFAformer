"""
Function:
Transformer-based attention mechanism for time series data.
date: 2025/06/11
Author: Qitong Chen and HuaYin
"""
import torch
from torch import nn
from einops import rearrange, repeat


# helpers


class Attention_single_domain_tf(nn.Module):
    """
    Transformer encoder block for time series data in a single domain with time, time-frequency and frequency embeddings.
    将信号的时域embedding、时频域embedding、频域embedding输入到Transformer编码器中。
    """
    def __init__(self, dim, heads=2, dim_head=64, dropout=0., args=None):
        super().__init__()
        inner_dim = dim_head * heads  # 计算多头注意力中每个样本的总特征维度。
        project_out = not (heads == 1 and dim_head == dim)  # 如果头数为1且每个头的维度等于输入维度，则不需要投影输出。
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，用于防止点积注意力中的数值过大。缩放 q 和 k 的点积，防止梯度爆炸。
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 一次性把输入变换为 Q、K、V 三个矩阵（拼在一起然后拆分）。
        self.operating = args.transformer_operating  # add_in_dots  add_in_attn  add_in_v  add_in_out
        # cat_in_dots  cat_in_attn  cat_in_v  cat_in_out
        self.map_rate = 3  # 多源域：3
        print(f'operating: {self.operating}, map_rate: {self.map_rate}')
        if self.operating == 'cat_in_v':
            self.linear_cat = nn.Sequential(
                nn.Linear((args.output_dim // 2) * 3, ((args.output_dim // 2) * self.map_rate)),
                nn.ReLU(),
                nn.Linear((args.output_dim // 2) * self.map_rate, args.output_dim // 2)
            )
        else:
            self.linear_cat = nn.Sequential(  # 用于将三个域的点积结果拼接后投影到原始维度。
                nn.Linear((args.n_patches + 1) * 3, (args.n_patches + 1) * self.map_rate),  # n_patches+1是因为包含了一个额外的分类token。
                # nn.ReLU(),
                # nn.GELU(),
                nn.Linear((args.n_patches + 1) * self.map_rate, (args.n_patches + 1))
            )
        # 多头注意力输出后需要重新投影回原维度，除非不变。
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_t_embedding, x_tf_embedding, x_f_embedding):
        # =====================================计算信号时域Embedding的多头自注意力=====================================
        x_t_norm = self.norm(x_t_embedding)
        qkv_t = self.to_qkv(x_t_norm).chunk(3, dim=-1)  # 将x投影变换为 Q、K、V（同时投影然后拆分为三个矩阵拆）
        q_t, k_t, v_t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_t)
        dots_t = torch.matmul(q_t, k_t.transpose(-1, -2)) * self.scale  # 计算 Q 和 K 的点积，得到注意力分数。
        attn_soft_t = self.softmax(dots_t)  # 对点积结果进行 softmax 归一化，得到注意力权重。
        attn_drop_t = self.dropout(attn_soft_t)  # 对注意力权重进行 dropout，防止过拟合。
        out1_t = torch.matmul(attn_drop_t, v_t)  # 将注意力权重与 V 相乘，得到加权后的输出。(64, 2, 129, 129)*(64, 2, 129, 128)=(64, 2, 129, 128)
        out2_t = rearrange(out1_t, 'b h n d -> b n (h d)')  # 将输出的维度重新排列为 (batch, n, heads * dim_head)，方便后续处理。
        out_t = self.to_out(out2_t)  # 将输出投影回原始维度，得到最终的输出。（64，129，256）
        # =====================================计算信号频域Embedding的多头自注意力===================================
        x_f_norm = self.norm(x_f_embedding)
        qkv_f = self.to_qkv(x_f_norm).chunk(3, dim=-1)
        q_f, k_f, v_f = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_f)
        dots_f = torch.matmul(q_f, k_f.transpose(-1, -2)) * self.scale
        attn_soft_f = self.softmax(dots_f)
        attn_drop_f = self.dropout(attn_soft_f)
        out1_f = torch.matmul(attn_drop_f, v_f)
        out2_f = rearrange(out1_f, 'b h n d -> b n (h d)')
        out_f = self.to_out(out2_f)
        # =====================================计算信号时频域Embedding的多头交叉注意力===================================
        x_tf_norm = self.norm(x_tf_embedding)
        qkv_tf = self.to_qkv(x_tf_norm).chunk(3, dim=-1)
        q_tf, k_tf, v_tf = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_tf)
        # =======================选择不同的操作方式来处理时域、时频域和频域的注意力计算========================
        if self.operating == 'cat_in_v':
            # 在 V 的计算中将三个域的 V 拼接。
            dots_tf1 = torch.matmul(q_t, k_f.transpose(-1, -2)) * self.scale
            dots_tf2 = torch.matmul(q_f, k_t.transpose(-1, -2)) * self.scale
            dots_tf3 = torch.matmul(q_tf, k_tf.transpose(-1, -2)) * self.scale
            attn_soft_tf1 = self.softmax(dots_tf1)
            attn_soft_tf2 = self.softmax(dots_tf2)
            attn_soft_tf3 = self.softmax(dots_tf3)
            attn_drop_tf1 = self.dropout(attn_soft_tf1)
            attn_drop_tf2 = self.dropout(attn_soft_tf2)
            attn_drop_tf3 = self.dropout(attn_soft_tf3)
            out1_tf1 = torch.matmul(attn_drop_tf1, v_tf)
            out1_tf2 = torch.matmul(attn_drop_tf2, v_tf)
            out1_tf3 = torch.matmul(attn_drop_tf3, v_tf)
            out1_tf_cat = torch.cat((out1_tf1, out1_tf2, out1_tf3), dim=-1)
            out1_tf = self.linear_cat(out1_tf_cat)  # 将拼接后的 V 投影到原始维度。
            out2_tf = rearrange(out1_tf, 'b h n d -> b n (h d)')
            out_tf = self.to_out(out2_tf)
        else:
            dots_tf = torch.matmul(q_t, k_f.transpose(-1, -2)) * self.scale
            attn_soft_tf = self.softmax(dots_tf)
            attn_drop_tf = self.dropout(attn_soft_tf)
            out1_tf = torch.matmul(attn_drop_tf, v_tf)
            out2_tf = rearrange(out1_tf, 'b h n d -> b n (h d)')
            out_tf = self.to_out(out2_tf)

        x_t_out = out_t + x_t_embedding
        x_tf_out = out_tf + x_tf_embedding
        x_f_out = out_f + x_f_embedding

        x_t_out = self.norm(x_t_out)
        x_tf_out = self.norm(x_tf_out)
        x_f_out = self.norm(x_f_out)
        return x_t_out, x_tf_out, x_f_out


if __name__ == '__main__':
    DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    dim = 256
    heads = 2
    dim_head = dim // heads
    model = Attention_single_domain_tf(dim=dim, heads=heads, dim_head=dim_head, dropout=0.).to(DEVICE)
    x_t = torch.randn(32, 128, dim).to(DEVICE)  # (batch_size, sequence_length（token 数）, feature_dim)
    x_tf = torch.randn(32, 128, dim).to(DEVICE)  # (batch_size, sequence_length（token 数）, feature_dim)
    x_f = torch.randn(32, 128, dim).to(DEVICE)  # (batch_size, sequence_length（token 数）, feature_dim)
    out_t, out_tf, out_f = model(x_t=x_t, x_tf=x_tf, x_f=x_f)
    print("Output shapes:")
    print("out_t:", out_t.shape)  # 应该是 (32, 128, dim)
    print("out_tf:", out_tf.shape)  # 应该是 (32, 128, dim)
    print("out_f:", out_f.shape)  # 应该是 (32, 128, dim)
