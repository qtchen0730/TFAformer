"""
Function:
1. 对于同一时间序列或同一频率序列的样本不再直接拼接patch_size_C个时间或频率戳，而是只拼接一个时间戳和一个频率戳。
2. 当时间和频率信息都需要嵌入时，时间戳和频率戳需要先进行linear和silu变换，然后进行拼接。
3. 返回时间幅值patch、频率幅值patch、时间-频率幅值patch。
date: 2025/06/07
Author: Qitong Chen and QiLi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import argparse
from argparse import Namespace
import numpy as np
from sklearn.preprocessing import StandardScaler


class E_01_HSE2_return_TF(nn.Module):
    def __init__(self, args):
        super(E_01_HSE2_return_TF, self).__init__()
        self.patch_size_L = args.patch_size_L
        self.patch_size_C = args.patch_size_C
        self.num_patches = args.n_patches
        self.output_dim = args.output_dim
        self.fs = args.fs
        self.emd_method = 'time_and_AmpValue_and_freqAmpValue'  # time_and_AmpValue_and_freqAmpValue
        self.rate = 1  # 线性变换的倍数 多源域时为1，单源域时为2
        # linear1 输入维度为x的（2+2）倍【时间信息嵌入（x，t）+频域信息嵌入拼接（x_f, freq）】
        if self.emd_method == 'timeAmpValue_and_freqAmpValue':  # （时序信号幅值，频域信号幅值）
            self.x_patch_linear = nn.Linear(self.patch_size_L * self.patch_size_C, self.patch_size_L * self.patch_size_C * self.rate)
            self.x_fft_patch_linear = nn.Linear(self.patch_size_L * self.patch_size_C, self.patch_size_L * self.patch_size_C * self.rate)
            self.linear_tf = nn.Linear(self.patch_size_L * self.patch_size_C * 2 * self.rate, self.output_dim)
            self.linear_t_out = nn.Linear(self.patch_size_L * self.patch_size_C * self.rate, self.output_dim)
            self.linear_f_out = nn.Linear(self.patch_size_L * self.patch_size_C * self.rate, self.output_dim)
        elif self.emd_method == 'time_and_AmpValue_and_freqAmpValue':  # （时间信息，时序信号幅值，频域信号幅值）
            self.x_patch_linear = nn.Linear(self.patch_size_L * (self.patch_size_C + 1), self.patch_size_L * (self.patch_size_C + 1) * self.rate)
            self.x_fft_patch_linear = nn.Linear(self.patch_size_L * (self.patch_size_C + 0), self.patch_size_L * (self.patch_size_C + 0) * self.rate)
            self.linear_tf = nn.Linear(self.patch_size_L * (self.patch_size_C * 2 + 1) * self.rate, self.output_dim)
            self.linear_t_out = nn.Linear(self.patch_size_L * (self.patch_size_C + 1) * self.rate, self.output_dim)
            self.linear_f_out = nn.Linear(self.patch_size_L * (self.patch_size_C + 0) * self.rate, self.output_dim)
        else:  # （时间信息，时序信号幅值，频率信息，频域信号幅值）
            self.x_patch_linear = nn.Linear(self.patch_size_L * (self.patch_size_C + 1), self.patch_size_L * (self.patch_size_C + 1) * self.rate)
            self.x_fft_patch_linear = nn.Linear(self.patch_size_L * (self.patch_size_C + 1), self.patch_size_L * (self.patch_size_C + 1) * self.rate)
            self.linear_tf = nn.Linear(self.patch_size_L * self.rate * (self.patch_size_C + 1) * 2, self.output_dim)
            self.linear_t_out = nn.Linear(self.patch_size_L * (self.patch_size_C + 1) * self.rate, self.output_dim)
            self.linear_f_out = nn.Linear(self.patch_size_L * (self.patch_size_C + 1) * self.rate, self.output_dim)

        self.linear_tf_out = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x[:, :, :]  # (B, L, C)
        B, L, C = x.size()
        device = x.device
        fs = self.fs
        T = 1.0 / fs

        # 构造时间向量
        t = torch.arange(L, device=device, dtype=torch.float32) * T
        t = t.unsqueeze(0).expand(B, -1)

        # 补齐
        if self.patch_size_L > L:
            rL = (self.patch_size_L + L - 1) // L
            x = repeat(x, 'b l c -> b (l r) c', r=rL)
            t = repeat(t, 'b l -> b (l r)', r=rL)
            L = x.size(1)

        if self.patch_size_C > C:
            rC = (self.patch_size_C + C - 1) // C
            x = repeat(x, 'b l c -> b l (c r)', r=rC)
            C = x.size(2)

        # ----- 频域变换 -----
        # ================================================ 频率求取并嵌入 ================================================
        fft_x = torch.fft.rfft(x, dim=1)  # 沿时间轴对输入做FFT，得到复数频谱，形状(B, L//2+1, C)
        fft_amp_value = torch.abs(fft_x)  # 取幅值，形状不变，形状(B, L//2+1, C)
        # 将频域数据插值为与时间域等长，用于拼接
        fft_amp_value_liner = F.interpolate(fft_amp_value.permute(0, 2, 1), size=L, mode='linear',
                                            align_corners=False).permute(0, 2, 1)  # 幅值插值后形状(B, L, C)
        # ================================================ 频率求取并嵌入end ==============================================
        # ----- 随机 patch 索引构造 -----
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        start_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)
        start_C = torch.randint(0, max_start_C + 1, (B, self.num_patches), device=device)
        # 这些 offsets 是用于在每个随机起点位置上提取一个 patch 的“模具”。上面的起始索引会加上这些 offsets，从而得到每个 patch 的完整索引。
        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)

        idx_L0 = (start_L.unsqueeze(-1) + offsets_L) % L  # (B, P, L_)
        idx_C0 = (start_C.unsqueeze(-1) + offsets_C) % C
        # 把 idx_L 和 idx_C 扩展成四维，是为了构造每个 patch 的“二维网格坐标（L × C）”，从而告诉 PyTorch 在每个样本里每个 patch 具体要取哪几个点。
        idx_L = idx_L0.unsqueeze(-1)  # (B, num_patches, patch_size_L, 1)，为patch增加通道维度，便于扩展通道索引
        idx_C = idx_C0.unsqueeze(-2)  # (B, num_patches, 1, patch_size_C)，为patch增加patch_size_L维度，便于扩展patch_size_L索引
        # ----- Patch 提取：时域 -----
        x_expand = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)  # 扩展为(B, num_patches, L, C)
        # 第二步：在时间维上选 patch 的时间段，现在要在第 2 维（时间轴）上挑选出长度为 patch_size_L 的时间点
        # idx_L 是时间上的索引，shape是 (B, num_patches, patch_size_L, 1)，扩展到(B, num_patches, patch_size_L, C)，每个时间点要取所有通道。
        x_patch0 = x_expand.gather(2, idx_L.expand(-1, -1, -1, C))
        # 第三步：在通道维上选 patch 的通道段，现在在第 3 维（通道轴）上挑出 patch_size_C 个通道
        # idx_C是通道上的索引，shape 是(B, num_patches, 1, patch_size_C)，扩展到(B,num_patches,patch_size_L,patch_size_C)，跟前面的时间维对齐
        # 即从每个 patch 的 (patch_size_L × C) 里再抽取出需要的通道 → 形成真正的 patch。
        x_patch = x_patch0.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))
        # ----- 时间嵌入 -------------
        t_expand = t.unsqueeze(1).expand(-1, self.num_patches, -1)
        t_patch0 = t_expand.gather(2, idx_L.squeeze(-1))
        t_patch = t_patch0.unsqueeze(-1)  # Revisions------------>t_patch0.unsqueeze(-1)
        # ----- Patch 提取：频域 -----
        x_fft_expand = fft_amp_value_liner.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        x_fft_patch0 = x_fft_expand.gather(2, idx_L.expand(-1, -1, -1, C))
        x_fft_patch = x_fft_patch0.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))
        if self.emd_method == 'time_and_AmpValue_and_freqAmpValue':  # （时间信息，时序信号幅值，频域信号幅值）
            x_patch_cat = torch.cat([x_patch, t_patch], dim=-1)
            x_patch_flat = rearrange(x_patch_cat, 'b p l c -> b p (l c)')
            x_patch_flat_linear = self.x_patch_linear(x_patch_flat)  # (B, P, L * C * rate)
            x_patch_flat_linear_silu = F.silu(x_patch_flat_linear)  # (B, P, L * C * rate)
            t_out = self.linear_t_out(x_patch_flat_linear_silu)  # (B, P, L * C * rate)
            freq_patch_flat = rearrange(x_fft_patch, 'b p l c -> b p (l c)')
            freq_patch_flat_linear = self.x_fft_patch_linear(freq_patch_flat)  # (B, P, L * C * rate)
            freq_patch_flat_linear_silu = F.silu(freq_patch_flat_linear)  # (B, P, L * C * rate)
            f_out = self.linear_f_out(freq_patch_flat_linear_silu)  # (B, P, L * C * rate)
            tf_patch = torch.cat([x_patch_flat_linear_silu, freq_patch_flat_linear_silu], dim=-1)
            tf_out1 = self.linear_tf(tf_patch)
            tf_out2 = F.silu(tf_out1)
            tf_out = self.linear_tf_out(tf_out2)
        

        return t_out, tf_out, f_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size_L', type=int, default=72)  # 故障类别数
    parser.add_argument('--patch_size_C', type=int, default=2)  # 学习率
    parser.add_argument('--n_patches', type=int, default=128)  # DAN损失的权衡参数
    parser.add_argument('--output_dim', type=int, default=160)  # 每次输入模型的样本数
    parser.add_argument('--fs', type=int, default=1600)  # 每次输入模型的样本数
    args = parser.parse_args()

    x = torch.randn((32, 1024, 6))
    embedding = E_01_HSE2_return_TF(args=args)
    t_out, tf_out, f_out = embedding(x)