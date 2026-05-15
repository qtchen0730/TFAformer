"""
Function:
Transformer model for time series data.
date: 2025/06/11
Author: Qitong Chen and HuaYin
"""
import torch
from torch import nn
from einops import rearrange, repeat
from models.TFAformer_block_time_series import Transformer_block_single_domain_tf_mul_layer
from Embedding.MAPE import E_01_HSE2_return_TF
import argparse
from argparse import Namespace
import torch.nn.functional as F



class Time_series_transformer_Mul_domain_tf(nn.Module):
    """
    Transformer encoder for time series data in a single source domain with time-frequency embedding.
    将单个域的数据x输入，首先对x求取时域、时频域和频率域的embedding：x_t_embedding, x_tf_embedding, x_f_embedding，
    然后将这三个embedding输入到Transformer编码器中，x_t_embedding求取自注意力, x_f_embedding求取自主意力，
    [x_t_embedding, x_tf_embedding, x_f_embedding]求取交叉注意力。
    """
    def __init__(self, *, num_classes, depth, heads, pool='cls', channels=1,
                 dropout=0., emb_dropout=0., args=None):
        super().__init__()
        num_patches = channels    # Channel数  《====================================================================
        patch_dim = args.output_dim    # 信号序列长度  《==============================================================
        dim = patch_dim  # 1024
        mlp_dim = 2*dim  # 2*dim
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.dim_head = args.output_dim // heads
        # self.transformer = Transformer_block_single_domain_tf(dim, depth, heads, self.dim_head, mlp_dim, dropout, args)
        self.transformer = Transformer_block_single_domain_tf_mul_layer(dim, depth, heads, self.dim_head, mlp_dim, dropout, args)

        self.pool = pool
        self.to_latent = nn.Identity()
        # ======================= 三个独立的分类头 ==========================
        if args.multi_head_classification is True:
            self.hidden_number = 4
            self.mlp_head_t = nn.Sequential(
                nn.Linear(dim, dim//self.hidden_number),
                nn.ReLU(inplace=True),
                nn.Linear(dim//self.hidden_number, num_classes)
            )
            self.mlp_head_tf = nn.Sequential(
                nn.Linear(dim, dim//self.hidden_number),
                nn.ReLU(inplace=True),
                nn.Linear(dim//self.hidden_number, num_classes)
            )
            self.mlp_head_f = nn.Sequential(
                nn.Linear(dim, dim//self.hidden_number),
                nn.ReLU(inplace=True),
                nn.Linear(dim//self.hidden_number, num_classes)
            )
        else:
            self.mlp_head = nn.Linear(dim, num_classes)

        self.norm = nn.LayerNorm(dim)
        self.args = args

        print("Using return_TF embedding method for time-frequency embedding.")
        self.tf_embedding = E_01_HSE2_return_TF(args=args)


    def forward(self, x):
        x_t_embedding, x_tf_embedding, x_f_embedding = self.tf_embedding(x)

        # ==========================================patch嵌入、类嵌入、位置嵌入=======================================
        # ========================patch嵌入===============================
        # =========================类嵌入=================================
        b, n, _ = x_t_embedding.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_t_cls_embedding = torch.cat((cls_tokens, x_t_embedding), dim=1)  # [96,129,160]
        x_tf_cls_embedding = torch.cat((cls_tokens, x_tf_embedding), dim=1)
        x_f_cls_embedding = torch.cat((cls_tokens, x_f_embedding), dim=1)
        # ========================位置嵌入==================================
        x_t_pos_embedding = x_t_cls_embedding + self.pos_embedding[:, :(n + 1)]  # [96,129,160]
        x_tf_pos_embedding = x_tf_cls_embedding + self.pos_embedding[:, :(n + 1)]
        x_f_pos_embedding = x_f_cls_embedding + self.pos_embedding[:, :(n + 1)]
        x_t_pos_embedding_drop = self.dropout(x_t_pos_embedding)  # [96, 129, 160]
        x_tf_pos_embedding_drop = self.dropout(x_tf_pos_embedding)
        x_f_pos_embedding_drop = self.dropout(x_f_pos_embedding)
        # ===================================*****Transformer编码器*****=========================================
        # x = self.norm(x)
        # y = self.norm(y)
        x_t_attention, x_tf_attention, x_f_attention = self.transformer(x_t_pos_embedding_drop, x_tf_pos_embedding_drop,
                                                                        x_f_pos_embedding_drop)  # [96, 129, 160]
        # ==========================================池化=========================================================
        # =================================时间embeddings输出======================================
        x_t_attention_mean = x_t_attention.mean(dim=1) if self.pool == 'mean' else x_t_attention[:, 0]    # [96, 160]
        x_t_attention_mean_latent = self.to_latent(x_t_attention_mean)    # [96, 160]
        if self.args.multi_head_classification is True:
            x_t_out = self.mlp_head_t(x_t_attention_mean_latent)  # [batch_size, num_class]
        else:
            x_t_out = self.mlp_head(x_t_attention_mean_latent)
        # ================================时频embeddings输出=======================================
        x_tf_attention_mean = x_tf_attention.mean(dim=1) if self.pool == 'mean' else x_tf_attention[:, 0]
        x_tf_attention_mean_latent = self.to_latent(x_tf_attention_mean)
        if self.args.multi_head_classification is True:
            x_tf_out = self.mlp_head_tf(x_tf_attention_mean_latent)
        else:
            x_tf_out = self.mlp_head(x_tf_attention_mean_latent)
        # ================================频率embeddings输出=======================================
        x_f_attention_mean = x_f_attention.mean(dim=1) if self.pool == 'mean' else x_f_attention[:, 0]
        x_f_attention_mean_latent = self.to_latent(x_f_attention_mean)
        if self.args.multi_head_classification is True:
            x_f_out = self.mlp_head_f(x_f_attention_mean_latent)
        else:
            x_f_out = self.mlp_head(x_f_attention_mean_latent)
        
        return x_t_out, x_tf_out, x_f_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=10)  # 10 or 4
    parser.add_argument('--patch_size_L', type=int, default=64)  # 64
    parser.add_argument('--patch_size_C', type=int, default=2)  # 3
    parser.add_argument('--n_patches', type=int, default=128)  # 256
    parser.add_argument('--output_dim', type=int, default=256)  # 256
    parser.add_argument('--fs', type=int, default=1600)
    parser.add_argument('--paper_type', type=str, default='Journal')
    parser.add_argument('--transformer_operating', type=str, default='cat_in_attn')

    args = parser.parse_args()

    DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    Transformer_model = Time_series_transformer_single_domain_tf(num_classes=args.num_class,
                                                                 depth=2, heads=2, dropout=0.1, emb_dropout=0.1,
                                                                 channels=args.n_patches, args=args).to(DEVICE)
    x = torch.randn(64, 6, 1024).to(DEVICE)  # (batch_size, seq_len, dim)
    x_t_out, x_tf_out, x_f_out = Transformer_model(x)
    print("x_t_out shape:", x_t_out.shape)
    print("x_tf_out shape:", x_tf_out.shape)
    print("x_f_out shape:", x_f_out.shape)
    total = sum([param.nelement() for param in Transformer_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
