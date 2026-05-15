"""
Author: Qitong Chen
Function：
1.三个分支分别使用三个分类头进行分类
2.不使用样本自适应归一化
3.交叉注意力分支在QKV计算完成后再进行拼接（cat_in_v）
4.Embedding模块部分时域和频域部分均添加序列信息
Date：2025.11.6
Watch: 保存通过验证集获得的最佳准确率val_best_acc
"""

import numpy as np
import torch
import pandas as pd
import torch.optim as optim
from models.Multimodal_model import Multimodal_model_Mul_domain_tf
import os

from data_loader.dataloader import data_load_time_series
from utils_dsbn.save_other_functions import save_his, train_history, save_predict_labels
from argparse import Namespace
from dataset.file_path import datasets_path
import time
from tqdm import tqdm
import argparse
import random



"""
Paper5:
1.learning_rates: 1e-4
2.dataset_names: ['SCARA_multi_modal_datasets', 'PU_datasets_4classes_4096_1samples']
3.Transformer_model_time_series: self.tf_embedding = E_01_HSE2_return_TF(args=args, args_d=args_d)
4.Transformer_attention_time_series: self.operating = 'cat_in_attn'
"""
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rates', type=list, default=[3e-4],
                    help='[5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]')

parser.add_argument('--best_learning_rate', type=float, default=3e-10, help='模型最后使用的最佳学习率')
parser.add_argument('--alpha1_params', type=list, default=['TFAformer'], help='weight parameters')
parser.add_argument('--dataset_names', type=list, default=['PU_datasets_4classes_4096_1samples', 'BJUT_WT_datasets_IFAC'],
                    help='SCARA_multi_modal_datasets, PU_datasets_4classes_4096_1samples, BJUT_WT_datasets_IFAC')
parser.add_argument('--SCARA_transfer_tasks', type=list, default=[0, 3, 6, 9])  # [0, 3, 6, 9]
parser.add_argument('--PU_transfer_tasks', type=list, default=[0, 1, 2, 3])  # [0, 1, 2, 3]
parser.add_argument('--BJUT_transfer_tasks', type=list, default=[20, 25, 30, 35])  # [20, 25, 30, 35]
parser.add_argument('--HUST_transfer_tasks', type=list, default=[5, 10, 20, 30])  # [5, 10, 20, 30]
parser.add_argument('--transfer_tasks', type=list, default=[])
parser.add_argument('--repeat_times', type=int, default=5, help='5')  # 重复实验次数
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--warm_epoch', type=int, default=10)
parser.add_argument('--save_all_models', type=bool, default=False, help='False, True')  # 是否保存每次重复实验的模型
parser.add_argument('--path1', type=str, default='./save_dir/IFAC_Time_frequency/',
help='Time_frequency_aware_Transformer_without_cat_attn')

parser.add_argument('--transformer_operating', type=str, default='cat_in_v', help='Transformer operating mode')  # 暂时不用
parser.add_argument('--decision_domain', type=str, default='Time_TimeAndFreq_Freq', help='Time_TimeAndFreq_Freq, Only_Time, Only_Time_Freq, Only_Freq')  # 4  总通道4  best:4
parser.add_argument('--batch_size', type=int, default=32)  # 32
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--ele_frequency', type=str, default='1600Hz')
parser.add_argument('--vib_frequency', type=str, default='1600Hz')
parser.add_argument('--speeds', type=str, default='4000r')
parser.add_argument('--patch_size_L', type=int, default=72)  # 五通道电信号+xy通道振动 72
parser.add_argument('--patch_size_C', type=int, default=0)  # 5
parser.add_argument('--SCARA_patch_size_C', type=int, default=7)  # 5  总通道7
parser.add_argument('--PU_patch_size_C', type=int, default=3)  # 2  总通道3
parser.add_argument('--BJUT_patch_size_C', type=int, default=3)  # 2  总通道3
parser.add_argument('--HUST_patch_size_C', type=int, default=4)  # 4  总通道4
parser.add_argument('--n_patches', type=int, default=128)  # 128
parser.add_argument('--output_dim', type=int, default=160)  # 160
parser.add_argument('--fs', type=int, default=0, help='sampling frequency')
parser.add_argument('--SCARA_fs', type=int, default=1600, help='SCARA sampling frequency')
parser.add_argument('--PU_fs', type=int, default=64000, help='PU sampling frequency')
parser.add_argument('--BJUT_fs', type=int, default=48000, help='BJUT sampling frequency')
parser.add_argument('--HUST_fs', type=int, default=25600, help='HUST sampling frequency')
parser.add_argument('--norm_weight', type=float, default=0, help='sample norm weight')

parser.add_argument('--multi_head_classification', type=bool, default=True,
                    help='True, False')
parser.add_argument('--embedding_methods', type=str, default='MAPE', help='cnn, MAPE', )  # 是否保存每次重复实验的模型
parser.add_argument('--n_channels', type=int, default=0, help='number_channels')  # 是否保存每次重复实验的模型
parser.add_argument('--SCARA_n_channels', type=int, default=7)  # 5  总通道7  best:7
parser.add_argument('--PU_n_channels', type=int, default=3)  # 2  总通道3  best:3
parser.add_argument('--BJUT_n_channels', type=int, default=3)  # 2  总通道3   best:2
parser.add_argument('--HUST_n_channels', type=int, default=4)  # 4  总通道4  best:4



args = parser.parse_args()
DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


def model_test(model, dataloader_src1_train, dataloader_src1_val, dataloader_src2_train, dataloader_src2_val,
         dataloader_src3_train, dataloader_src3_val, dataloader_tar):
    """
    Test the model on source and target datasets.
    Args:
        model: The model to be tested.
        dataloader_src1_train: 加载的源域1训练集
        dataloader_src1_val:  加载的源域1验证集
        dataloader_src2_train:  加载的源域2训练集
        dataloader_src2_val:  加载的源域2验证集
        dataloader_src3_train:  加载的源域3训练集
        dataloader_src3_val:  加载的源域3验证集
        dataloader_tar:  加载的目标域数据集
    Returns: 测试结果

    """
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        # =============================Predict model on source domain training set====================================
        n_samples_src_train = 0  # 源域训练集样本数
        n_samples_src_train_correct = 0  # 源域训练集正确预测样本数
        src_train_loss_per_epoch = []  # 源域训练集损失
        for (src1_train_data, src2_train_data, src3_train_data) in zip(
                dataloader_src1_train, dataloader_src2_train, dataloader_src3_train):
            x_src1_train, y_src1_train = src1_train_data
            x_src2_train, y_src2_train = src2_train_data
            x_src3_train, y_src3_train = src3_train_data
            x_src_train = torch.cat((x_src1_train, x_src2_train, x_src3_train), dim=0).to(DEVICE)
            y_src_train = torch.cat((y_src1_train, y_src2_train, y_src3_train), dim=0).to(DEVICE)
            # =============================Predict model============================================
           
            ys_train_time_pre, ys_train_time_freq_pre, ys_train_freq_pre = model(x_src_train)
            if args.decision_domain == 'Time_TimeAndFreq_Freq':
                ys_train_pre = (ys_train_time_pre + ys_train_time_freq_pre + ys_train_freq_pre) / 3
            elif args.decision_domain == 'Only_Time':
                ys_train_pre = ys_train_time_pre  # Time loss
            elif args.decision_domain == 'Only_Time_Freq':
                ys_train_pre = ys_train_time_freq_pre  # Time_frequency loss
            elif args.decision_domain == 'Only_Freq':
                ys_train_pre = ys_train_freq_pre  # Frequency loss
            src_train_pre_label = torch.argmax(ys_train_pre, 1)
            # =============================Calculate accuracy ========================================
            n_samples_src_train += len(y_src_train)
            n_samples_src_train_correct += (src_train_pre_label == y_src_train.long()).sum().item()
            # =============================Calculate loss ============================================
            loss_s_train = loss_fn(ys_train_pre, y_src_train)
            src_train_loss_per_epoch.append(loss_s_train.item())
        # =============================Predict model on source domain validation set==================================
        n_samples_src_val = 0  # 源域验证集样本数
        n_samples_src_val_correct = 0  # 源域验证集正确预测样本数
        src_val_loss_per_epoch = []  # 源域验证集损失
        for (src1_val_data, src2_val_data, src3_val_data) in zip(dataloader_src1_val, dataloader_src2_val, dataloader_src3_val):
            x_src1_val, y_src1_val = src1_val_data
            x_src2_val, y_src2_val = src2_val_data
            x_src3_val, y_src3_val = src3_val_data
            x_src_val = torch.cat((x_src1_val, x_src2_val, x_src3_val), dim=0).to(DEVICE)
            y_src_val = torch.cat((y_src1_val, y_src2_val, y_src3_val), dim=0).to(DEVICE)
            # =============================Predict model============================================
            
            ys_val_time_pre, ys_val_time_freq_pre, ys_val_freq_pre = model(x_src_val)
            if args.decision_domain == 'Time_TimeAndFreq_Freq':
                ys_val_pre = (ys_val_time_pre + ys_val_time_freq_pre + ys_val_freq_pre) / 3
            elif args.decision_domain == 'Only_Time':
                ys_val_pre = ys_val_time_pre  # Time loss
            elif args.decision_domain == 'Only_Time_Freq':
                ys_val_pre = ys_val_time_freq_pre  # Time_frequency loss
            elif args.decision_domain == 'Only_Freq':
                ys_val_pre = ys_val_freq_pre  # Frequency loss

            src_val_pre_label = torch.argmax(ys_val_pre, 1)
            # =============================Calculate accuracy ========================================
            n_samples_src_val += len(y_src_val)
            n_samples_src_val_correct += (src_val_pre_label == y_src_val.long()).sum().item()
            # =============================Calculate loss ============================================
            loss_s_val = loss_fn(ys_val_pre, y_src_val)
            src_val_loss_per_epoch.append(loss_s_val.item())
        # =============================Predict model on target domain data set========================================
        n_samples_tar = 0  # 目标域样本数
        n_samples_tar_correct = 0  # 目标域正确预测样本数
        tar_loss_per_epoch = []  # 目标域损失
        tar_test_label_list = []  # 目标域测试集标签列表
        tar_test_pre_label_list = []  # 目标域测试集预测标签列表
        for tar_data in dataloader_tar:
            x_tar, y_tar = tar_data
            x_tar, y_tar = x_tar.to(DEVICE), y_tar.to(DEVICE)
            # =============================Predict model============================================
            yt_time_pre, yt_time_freq_pre, yt_freq_pre = model(x_tar)
            if args.decision_domain == 'Time_TimeAndFreq_Freq':
                yt_pre = (yt_time_pre + yt_time_freq_pre + yt_freq_pre) / 3
            elif args.decision_domain == 'Only_Time':
                yt_pre = yt_time_pre  # Time loss
            elif args.decision_domain == 'Only_Time_Freq':
                yt_pre = yt_time_freq_pre  # Time_frequency loss
            elif args.decision_domain == 'Only_Freq':
                yt_pre = yt_freq_pre  # Frequency loss
            tar_pre_label = torch.argmax(yt_pre, 1)
            # =============================Calculate accuracy ========================================
            n_samples_tar += len(y_tar)
            n_samples_tar_correct += (tar_pre_label == y_tar.long()).sum().item()
            # =============================Calculate loss ============================================
            loss_tar = loss_fn(yt_pre, y_tar)
            tar_loss_per_epoch.append(loss_tar.item())
            # =============================Save target test label and prediction label ===============
            tar_test_label_list.append(y_tar)
            tar_test_pre_label_list.append(tar_pre_label)

    tar_test_labels = torch.cat(tar_test_label_list, axis=0)
    tar_test_pre_labels = torch.cat(tar_test_pre_label_list, axis=0)
    acc_src_train = float(n_samples_src_train_correct) / n_samples_src_train
    acc_src_val = float(n_samples_src_val_correct) / n_samples_src_val
    tar_acc = float(n_samples_tar_correct) / n_samples_tar
    src_train_loss_mean = np.mean(src_train_loss_per_epoch)
    src_val_loss_mean = np.mean(src_val_loss_per_epoch)
    tar_loss_mean = np.mean(tar_loss_per_epoch)
    model.train()
    return acc_src_train, src_train_loss_mean, acc_src_val, src_val_loss_mean, tar_acc, tar_loss_mean, tar_test_labels, tar_test_pre_labels


def train(model, optimizer, dataloader_src1_train, dataloader_src1_val, dataloader_src2_train, dataloader_src2_val,
          dataloader_src3_train, dataloader_src3_val, dataloader_tar, save_name='', save_all_models=False,
          repeat_times=None, train_hist=None):
    """
    Train the model with source datasets.
    Args:
        model: 模型
        optimizer: 优化器
        dataloader_src1_train: 源域1的训练集
        dataloader_src1_val: 源域1的验证集
        dataloader_src2_train: 源域2的训练集
        dataloader_src2_val: 源域2的验证集
        dataloader_src3_train: 源域3的训练集
        dataloader_src3_val: 源域3的验证集
        dataloader_tar: 目标域数据集
        save_name: 模型保存名称
        save_all_models: 是否保存运行10次的模型
        repeat_times: 当前重复实验次数，默认10次
        train_hist: 训练记录
    Returns:
    """
    loss_class = torch.nn.CrossEntropyLoss()

    best_acc = -float('inf')  # 目标域测试集最佳准确率
    val_best_acc = -float('inf')  # 验证集最佳准确率
    best_acc_val_best_acc_min_loss_acquire = -float('inf')  # 使用源域验证集最佳准确率和最小损失测试得到的最佳目标域测试集准确率
    val_min_loss = float('inf')
    src_time_loss_per_epoch = []
    src_time_freq_loss_per_epoch = []
    src_freq_loss_per_epoch = []
    src_total_loss_per_epoch = []
    for epoch in range(args.n_epoch):
        for (src1_train_data, src2_train_data, src3_train_data) in zip(
                dataloader_src1_train, dataloader_src2_train, dataloader_src3_train):
            x_src1_train, y_src1_train = src1_train_data
            x_src2_train, y_src2_train = src2_train_data
            x_src3_train, y_src3_train = src3_train_data
            x_src = torch.cat((x_src1_train, x_src2_train, x_src3_train), dim=0).to(DEVICE)
            y_src = torch.cat((y_src1_train, y_src2_train, y_src3_train), dim=0).to(DEVICE)
            # ================================== 三个域的数据拼接后一起输入模型 =========================================== #
            
            y_t_pre, y_tf_pre, y_f_pre = model(x_src)
            src_time_loss = loss_class(y_t_pre, y_src)
            src_time_freq_loss = loss_class(y_tf_pre, y_src)
            src_freq_loss = loss_class(y_f_pre, y_src)
            if args.decision_domain == 'Time_TimeAndFreq_Freq':
                total_loss = (src_time_loss + src_time_freq_loss + src_freq_loss) / 3  # Time-frequency cross loss
            elif args.decision_domain == 'Only_Time':
                total_loss = src_time_loss  # Time loss
            elif args.decision_domain == 'Only_Time_Freq':
                total_loss = src_time_freq_loss  # Time_frequency loss
            elif args.decision_domain == 'Only_Freq':
                total_loss = src_freq_loss  # Frequency loss
            # =============================backward and optimize======================
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            src_time_loss_per_epoch.append(src_time_loss.item())
            src_time_freq_loss_per_epoch.append(src_time_freq_loss.item())
            src_freq_loss_per_epoch.append(src_freq_loss.item())
            # src_time_loss_per_epoch.append(0)
            # src_time_freq_loss_per_epoch.append(0)
            # src_freq_loss_per_epoch.append(0)

            src_total_loss_per_epoch.append(total_loss.item())
        src_train_time_loss_per_epoch_mean = np.mean(src_time_loss_per_epoch)
        src_train_time_freq_loss_per_epoch_mean = np.mean(src_time_freq_loss_per_epoch)
        src_train_freq_loss_per_epoch_mean = np.mean(src_freq_loss_per_epoch)
        src_train_total_loss_per_epoch_mean = np.mean(src_total_loss_per_epoch)
        item_pr = 'Epoch: [{}/{}], src_time_loss: {:.4f}, src_time_freq_loss: {:.4f}, src_freq_loss: {:.4f}, ' \
                  'total_loss: {:.4f}'.format(
                   epoch+1, args.n_epoch, src_train_time_loss_per_epoch_mean, src_train_time_freq_loss_per_epoch_mean,
                   src_train_freq_loss_per_epoch_mean, src_train_total_loss_per_epoch_mean)
        print(item_pr, end=' >>> ')

        # =============================Validate and test model============================================
        src_test_acc, src_test_loss, src_val_acc, src_val_loss, tar_test_acc, tar_test_loss, tar_test_labels, tar_test_pre_labels = model_test(
            model=model, dataloader_src1_train=dataloader_src1_train, dataloader_src2_train=dataloader_src2_train,
            dataloader_src3_train=dataloader_src3_train, dataloader_src1_val=dataloader_src1_val,
            dataloader_src2_val=dataloader_src2_val, dataloader_src3_val=dataloader_src3_val, dataloader_tar=dataloader_tar)
        test_info = 'Source train acc: {:.3f} %, Source val acc: {:.3f} %, Target acc: {:.3f} %'.format(
            src_test_acc * 100, src_val_acc * 100, tar_test_acc * 100)
        print(test_info)
        # ============================Save model============================================
        if best_acc <= tar_test_acc:  # 使用目标域数据集直接测试来记录目标域最佳准确率
            best_acc = tar_test_acc
        if val_best_acc <= src_val_acc and src_val_loss <= val_min_loss:  # 使用源域验证集最佳准确率和最小损失来记录目标域最佳准确率
            val_best_acc = src_val_acc
            val_min_loss = src_val_loss
            best_acc_val_best_acc_min_loss_acquire = tar_test_acc
            if save_all_models:
                torch.save(model, '{}_val_best_acc_min_loss_acquire.pth'.format(save_name))
            else:
                if repeat_times == args.repeat_times - 1:
                    torch.save(model, '{}_val_best_acc.pth'.format(save_name))  # val best acc & min loss acquire
                else:
                    pass
            save_predict_labels(yt_label=tar_test_labels, yt_pre_label=tar_test_pre_labels,
                                save_name='{}_val_best_acc_prediction_labels.csv'.format(save_name))
        print('Best test acc: {:.3f} %, Best val acc: {:.3f} %, Min val loss : {:.5f},'
              'Best val acc and val min loss acquire test acc: {:.3f} %'.format(
               best_acc * 100, val_best_acc * 100, val_min_loss, best_acc_val_best_acc_min_loss_acquire * 100)
              )
        train_hist['src_train_time_loss'].append(src_train_time_loss_per_epoch_mean)
        train_hist['src_train_time_freq_loss'].append(src_train_time_freq_loss_per_epoch_mean)
        train_hist['src_train_freq_loss'].append(src_train_freq_loss_per_epoch_mean)
        train_hist['src_train_total_loss'].append(src_train_total_loss_per_epoch_mean)
        train_hist['src_test_acc'].append(src_test_acc)
        train_hist['src_test_loss'].append(src_test_loss)
        train_hist['src_val_acc'].append(src_val_acc)
        train_hist['src_val_loss'].append(src_val_loss)
        train_hist['tar_test_acc'].append(tar_test_acc)
        train_hist['tar_test_loss'].append(tar_test_loss)
        train_hist['val_acquire_best_acc'].append(best_acc_val_best_acc_min_loss_acquire)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    total_start_time = time.time()
    os.makedirs(args.path1, exist_ok=True)
    # 总记录文件夹，记录每个学习率、每个alpha1、每个数据集所有迁移任务重复执行10次的均值与标准差数据
    fp_val_best_acc_all_experiments_records = open(
        args.path1 + 'The mean and std deviation of val best acc from the all experiments.csv', 'a')
    # ===================================================遍历每个学习率超参数===============================================
    for learning_rate in tqdm(args.learning_rates, desc="Learning Rates Progress"):
        if learning_rate == args.best_learning_rate:
            args.save_all_models = True  # 最佳学习率时保存所有模型
        else:
            args.save_all_models = False
        learning_rate_str = 'lr_' + str(learning_rate)
        path2 = args.path1 + learning_rate_str + '/'
        os.makedirs(path2, exist_ok=True)
        fp_per_weight_records = open(path2 + 'The mean and std deviation of the weight experiments.csv', 'a')  # a不会擦除原记录
        # ===================================遍历每个alpha1超参数=========================================================
        for alpha1_param in tqdm(args.alpha1_params, desc="Alpha1 Params Progress"):
            alpha1_param_str = str(alpha1_param)
            args.norm_weight = alpha1_param
            path3 = path2 + alpha1_param_str + '/'
            os.makedirs(path3, exist_ok=True)
            fp_per_dataset_records = open(path3 + 'The mean and std deviation of the dataset experiments.csv', 'a')
            # ===================================遍历每个数据集=========================================================
            for dataset_name in tqdm(args.dataset_names, desc="Dataset Names Progress"):
                if dataset_name == 'SCARA_multi_modal_datasets':
                    args.transfer_tasks = args.SCARA_transfer_tasks  # SCARA_multi_modal_datasets
                    args.patch_size_C = args.SCARA_patch_size_C
                    args.fs = args.SCARA_fs  # SCARA_fs
                    args.n_channels = args.SCARA_n_channels
                elif dataset_name == 'PU_datasets_3classes_4096_1samples' or dataset_name == 'PU_datasets_4classes_4096_1samples':
                    args.transfer_tasks = args.PU_transfer_tasks
                    args.patch_size_C = args.PU_patch_size_C
                    args.fs = args.PU_fs  # PU_fs
                    args.n_channels = args.PU_n_channels
                elif dataset_name == 'BJUT_WT_datasets_IFAC':
                    args.transfer_tasks = args.BJUT_transfer_tasks
                    args.patch_size_C = args.BJUT_patch_size_C
                    args.fs = args.BJUT_fs  # BJUT_fs
                    args.n_channels = args.BJUT_n_channels

                path4 = path3 + dataset_name + '/'
                os.makedirs(path4, exist_ok=True)
                fp_mean_std_per_task = open(path4 + 'The mean and std deviation of per task.csv', 'a')  # a不会擦除原记录
                # ==================================遍历每个迁移任务=====================================================
                for transfer_task in tqdm(args.transfer_tasks, desc="Transfer Tasks Progress"):
                    transfer_task_str = 'T' + str(transfer_task)
                    data_root, src_dataset_name1, src_dataset_name2, src_dataset_name3, tgt_dataset_name, classes_number = datasets_path(
                        dataset_name=dataset_name, task_name=transfer_task_str, speeds=args.speeds,
                        ele_frequency=args.ele_frequency, vib_frequency=args.vib_frequency)
                    if dataset_name == 'SCARA_multi_modal_datasets':
                        save_name = src_dataset_name1[0:7] + src_dataset_name2[6:7] + src_dataset_name3[6:9] + '_to_' + tgt_dataset_name[6:]
                    elif dataset_name == 'HUST_motor_datasets':
                        save_name = src_dataset_name1[:10] + src_dataset_name2[5:10] + src_dataset_name3[5:10] + 'to_' + tgt_dataset_name[5:9]
                    else:
                        save_name = src_dataset_name1[:14] + src_dataset_name2[2:14] + src_dataset_name3[2:14] + '_to_' + tgt_dataset_name[3:14]
                    path5 = path4 + transfer_task_str + '/'
                    os.makedirs(path5, exist_ok=True)
                    val_best_acc_per_experimental_task = {}
                    val_best_acc_per_experimental_task['val_best_accuracy'] = []  # 为避免偶然性，每次任务跑10次或多次，记录每次跑完后的best_acc
                    val_best_acc_per_experimental_task['time_spend'] = []
                    # ==============================重复训练10次===================================================
                    for number in tqdm(range(args.repeat_times), desc="Repeat Times Progress"):
                        train_history_record = train_history()
                        start_time = time.time()
                        times_str = str(number + 1) + '/'
                        path6 = path5 + times_str
                        os.makedirs(path6, exist_ok=True)
                        # torch.random.manual_seed(number+1)
                        # set_seed(number + 1)
                        loader_src1_train, loader_src1_val, _ = data_load_time_series(
                            dataset_path=data_root + src_dataset_name1 + '_data.npy',
                            dataset_label_path=data_root + src_dataset_name1 + '_label.npy',
                            batch_size=args.batch_size, target_dataset=False, dataset_name=dataset_name,
                            classes_number=classes_number, seed=(number+1)
                        )
                        loader_src2_train, loader_src2_val, _ = data_load_time_series(
                            dataset_path=data_root + src_dataset_name2 + '_data.npy',
                            dataset_label_path=data_root + src_dataset_name2 + '_label.npy',
                            batch_size=args.batch_size, target_dataset=False, dataset_name=dataset_name,
                            classes_number=classes_number, seed=(number+1)
                        )
                        loader_src3_train, loader_src3_val, _ = data_load_time_series(
                            dataset_path=data_root + src_dataset_name3 + '_data.npy',
                            dataset_label_path=data_root + src_dataset_name3 + '_label.npy',
                            batch_size=args.batch_size, target_dataset=False, dataset_name=dataset_name,
                            classes_number=classes_number, seed=(number+1)
                        )
                        loader_tar, channels_number = data_load_time_series(
                            dataset_path=data_root + tgt_dataset_name + '_data.npy',
                            dataset_label_path=data_root + tgt_dataset_name + '_label.npy',
                            batch_size=args.batch_size, target_dataset=True, dataset_name=dataset_name,
                            classes_number=classes_number, seed=(number+1)
                        )
                        set_seed(number + 1)
                        model = Multimodal_model_Mul_domain_tf(class_num=classes_number, args=args,
                                                                  channels_number=channels_number).to(DEVICE)
                        model.to(DEVICE)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        train(model, optimizer, dataloader_src1_train=loader_src1_train,
                              dataloader_src1_val=loader_src1_val,
                              dataloader_src2_train=loader_src2_train, dataloader_src2_val=loader_src2_val,
                              dataloader_src3_train=loader_src3_train, dataloader_src3_val=loader_src3_val,
                              dataloader_tar=loader_tar, save_name=path6 + save_name,
                              save_all_models=args.save_all_models, repeat_times=number,
                              train_hist=train_history_record)
                        save_his(train_hist=train_history_record, save_name='{}_training_history.csv'.format(path6 + save_name))
                        best_accuracy = np.max(train_history_record['tar_test_acc'])
                        best_acc_val_best_acc_min_loss_acquire = train_history_record['val_acquire_best_acc'][-1]
                        print(
                            'Best test acc: {:.3f} %, The best accuracy obtained from the validation set:{:.3f} % '.format(
                                best_accuracy * 100, best_acc_val_best_acc_min_loss_acquire * 100))
                        val_best_acc_per_experimental_task['val_best_accuracy'].append(best_acc_val_best_acc_min_loss_acquire)  # 每次实验的best_acc
                        end_time = time.time()
                        time_spend = abs(start_time - end_time)
                        val_best_acc_per_experimental_task['time_spend'].append(time_spend)  # 每次实验的运行时间
                    # ==============================end======================================================
                        acc_str = str(number + 1) + '-' + '{:.5f}'.format(best_acc_val_best_acc_min_loss_acquire) + '/'
                        path6_new = path5 + acc_str
                        if os.path.exists(path6):
                            os.rename(path6, path6_new)
                        else:
                            print(f"文件夹'{path6}'Not Exist!!!'")
                    # =================保存每次实验的best_acc和运行时间======================
                    data_df = pd.DataFrame(val_best_acc_per_experimental_task)
                    data_df.to_csv(path5 + save_name + '_best_acc_and_total_time.csv')
                    # ==============计算每个迁移任务(10次实验)的best_acc均值和标准差============
                    best_acc_per_task_mean = np.mean(val_best_acc_per_experimental_task['val_best_accuracy'])  # 10次best_acc的均值
                    best_acc_per_task_std = np.std(val_best_acc_per_experimental_task['val_best_accuracy'], ddof=1)  # 标准差
                    time_per_task_mean = np.mean(val_best_acc_per_experimental_task['time_spend'])  # 10次的时间均值
                    time_per_task_var = np.std(val_best_acc_per_experimental_task['time_spend'], ddof=1)  # 10次的时间标准差
                    # =================每个迁移任务的best_acc均值和标准差记录==================
                    per_task_best_acc_mean_std = \
                        '{}, ' \
                        'Best_acc_per_task_mean:, {:.5f}, Best_acc_per_task_mean_var:, {:.2f}±{:.2f}, ' \
                        'time_per_task_mean:,{:.5f}, time_per_task_mean_var:,{:.2f}±{:.2f}'.format(
                            transfer_task_str,
                            best_acc_per_task_mean, best_acc_per_task_mean * 100, best_acc_per_task_std * 100,
                            time_per_task_mean, time_per_task_mean, time_per_task_var)
                    fp_mean_std_per_task.write(per_task_best_acc_mean_std + '\n')
                    # ==============每个数据集的实验记录，数据集名称-迁移任务====================
                    per_dataset_records = \
                        '{}, {}, ' \
                        'Best_acc_per_task_mean:, {:.5f}, Best_acc_per_task_mean_var:, {:.2f}±{:.2f},' \
                        'time_per_task_mean:,{:.5f}, time_per_task_mean_var:,{:.2f}±{:.2f}'.format(
                            dataset_name, transfer_task_str,
                            best_acc_per_task_mean, best_acc_per_task_mean * 100, best_acc_per_task_std * 100,
                            time_per_task_mean, time_per_task_mean, time_per_task_var)
                    fp_per_dataset_records.write(per_dataset_records + '\n')
                    # =============每个weight的实验记录，alpha权重-数据集名称-迁移任务===========
                    per_weight_records = \
                        '{}, {}, {}, ' \
                        'Best_acc_per_task_mean:, {:.5f}, Best_acc_per_task_mean_var:, {:.2f}±{:.2f},' \
                        'time_per_task_mean:,{:.5f}, time_per_task_mean_var:,{:.2f}±{:.2f}'.format(
                            alpha1_param_str, dataset_name, transfer_task_str,
                            best_acc_per_task_mean, best_acc_per_task_mean * 100, best_acc_per_task_std * 100,
                            time_per_task_mean, time_per_task_mean, time_per_task_var)
                    fp_per_weight_records.write(per_weight_records + '\n')
                    # ================总的实验记录，学习率、权重、数据集名称、迁移任务=============
                    all_experiments_records = \
                        '{}, {}, {}, {}, ' \
                        'Best_acc_per_task_mean:, {:.5f}, Best_acc_per_task_mean_var:, {:.2f}±{:.2f},' \
                        'time_per_task_mean:,{:.5f}, time_per_task_mean_var:,{:.2f}±{:.2f}'.format(
                            learning_rate_str, alpha1_param_str, dataset_name, transfer_task_str,
                            best_acc_per_task_mean, best_acc_per_task_mean * 100, best_acc_per_task_std * 100,
                            time_per_task_mean, time_per_task_mean, time_per_task_var)
                    fp_val_best_acc_all_experiments_records.write(all_experiments_records + '\n')
                fp_mean_std_per_task.close()  # 关闭每个迁移任务的记录文件
                fp_per_dataset_records.write('\n')  # 每个数据集的记录文件换行
                fp_per_weight_records.write('\n')
                fp_val_best_acc_all_experiments_records.write('\n')
            fp_per_dataset_records.close()  # 关闭每个数据集的记录文件
            fp_val_best_acc_all_experiments_records.write('\n')
            fp_per_weight_records.write('\n')
        fp_per_weight_records.close()  # 关闭每个权重的记录文件
        fp_val_best_acc_all_experiments_records.write('\n')
    fp_val_best_acc_all_experiments_records.close()  # 关闭总的实验记录文件

    total_end_time = time.time()
    all_task_total_time = abs(total_start_time - total_end_time)
    print('Total time of the all tasks training = {} s'.format(round(all_task_total_time, 3)))
