import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from models.TFAformer_model_time_series import  Time_series_transformer_Mul_domain_tf
import argparse
from argparse import Namespace



class Multimodal_model_Mul_domain_tf(nn.Module):
    """
    多模态time-frequency Transformer模型
    Args:
        class_num (int, optional): The number of classes for classification tasks. Defaults to None.
        args (Namespace, optional): Additional arguments for the model configuration. Defaults to None.
    """
    def __init__(self, class_num=None, args=None, channels_number=0):
        super(Multimodal_model_Mul_domain_tf, self).__init__()
        self.args = args
        
        self.sharedNetwork = Time_series_transformer_Mul_domain_tf(num_classes=class_num, depth=1, heads=2,
                                                                        dropout=0.1, emb_dropout=0.1,
                                                                        channels=args.n_patches, args=args,)

    def forward(self, x):
        x_t_out, x_tf_out, x_f_out = self.sharedNetwork(x)
        return x_t_out, x_tf_out, x_f_out


if __name__ == "__main__":
    data_name = 'a_013_SUDA'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=3)  # 10 or 4
    parser.add_argument('--patch_size_L', type=int, default=64)  # 64
    parser.add_argument('--patch_size_C', type=int, default=2)  # 3
    parser.add_argument('--n_patches', type=int, default=128)  # 256
    parser.add_argument('--output_dim', type=int, default=256)  # 256
    parser.add_argument('--fs', type=int, default=1600)
    parser.add_argument('--norm_weight', type=float, default=0, help='sample norm weight')
    parser.add_argument('--paper_type', type=str, default='Conference_IFAC', help='Conference, Conference_WT, Conference_IFAC, Journal')
    parser.add_argument('--transformer_operating', type=str, default='cat_in_v',  # cat_in_v  without_cat
                    help='[cat_in_attn, without_cat_in_attn]'
                         '[add_in_dots  add_in_attn  add_in_v  add_in_out cat_in_dots  '  # 暂时不用
                         'cat_in_attn  cat_in_v  cat_in_out]')  # 暂时不用
    parser.add_argument('--multi_head_classification', type=bool, default=True, help='True, False')
    parser.add_argument('--embedding_methods', type=str, default='MAPE', help='cnn, MAPE', )  # 是否保存每次重复实验的模型
    parser.add_argument('--adaptive_fusion', type=bool, default=False, help='False, True')  # 是否保存每次重复实验的模型
    parser.add_argument('--weight_fusion', type=bool, default=False, help='False, True')  # 是否保存每次重复实验的模型
    parser.add_argument('--distance_strategy', type=bool, default=False, help='True, False')


    args = parser.parse_args()

    DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    model = Multimodal_model_single_domain_tf(class_num=args.num_class, args=args).to(DEVICE)

    x = torch.randn(32, 1, 1024).to(DEVICE)  # (batch_size, seq_len, dim)
    x_t_out, x_tf_out, x_f_out = model(x)
    print("x_t_out shape:", x_t_out.shape)
    print("x_tf_out shape:", x_tf_out.shape)
    print("x_f_out shape:", x_f_out.shape)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    
    total_flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                x_t_out, x_tf_out, x_f_out = model(x)
            total_flops = fcm.get_total_flops()
    except Exception as exc:
        with torch.no_grad():
            x_t_out, x_tf_out, x_f_out = model(x)
        print(f"FLOPs calc failed: {exc}")
    if total_flops is not None:
        print(f"FLOPs (supported ops only): {total_flops:,}")
        print(f"GFLOPs (supported ops only): {total_flops / 1e9:.6f}")


