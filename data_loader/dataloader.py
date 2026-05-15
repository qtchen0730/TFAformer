# -*- coding: utf-8 -*-
"""
@Author       : 陈启通
@Time         : 2025年9月30日
@Desc         ：20250930 v1.0
@Descriptions ：加载源域与目标域数据
"""

from torch.utils.data import DataLoader, TensorDataset
import torch
from os.path import splitext
import scipy
import scipy.io as io
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer


def data_reader_fn(datadir, gpu=True):
    """
    read data from .mat
    Args:
        datadir: 加载数据文件
    """
    datatype = splitext(datadir)[1]
    if datatype == '.mat':

        data = scipy.io.loadmat(datadir)

        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
    if datatype == '':
        pass

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    y_train = torch.argmax(y_train, 1)  # dim=1 取表示行最大值
    y_test = torch.argmax(y_test, 1)
    return x_train, y_train, x_test, y_test


def dataload(batch_size=64, dataset_path=''):
    """
    加载源域数据集，即参与模型预训练的源域数据集
    :param batch_size: 每次加载样本数
    :param dataset_path: 数据集路径
    :return: loader好的序列数据集
    """
    x_train, y_train, x_test, y_test = data_reader_fn(dataset_path)

    torch_dataset = TensorDataset(x_train, y_train)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def test_dataload(batch_size=64, dataset_path=''):
    """
    加载目标域数据集，用于模型测试
    :param batch_size: 每次加载样本数
    :param dataset_path: 数据集路径
    :return: loader好的序列数据集
    """
    xt_train, yt_train, xt_test, yt_test = data_reader_fn(dataset_path)

    torch_dataset = TensorDataset(xt_test, yt_test)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def target_dataload(batch_size=64, target_data_name=''):
    t_dataset_path = 'C:/Users/CCSLab/Desktop/lixuan/HC_discrete_datasets_Full_cycle/Current_Feedback/'
    # t_dataset_path = 'C:/Users/CCSLab/Desktop/lixuan/轴承数据集/'

    tdatadir = t_dataset_path + target_data_name
    xt_train, yt_train, xt_test, yt_test = data_reader_fn(tdatadir)

    torch_dataset = TensorDataset(xt_test, yt_test)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def source_dataload(batch_size=64, source_data_name=''):
    s_dataset_path = 'D:/DeepLearning/dataset/fd/bearing_datasets/'
    sdatadir = s_dataset_path + source_data_name
    xs_train, ys_train, xs_test, ys_test = data_reader_fn(sdatadir)

    torch_dataset = TensorDataset(xs_test, ys_test)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def channel_wise_standardize(data, scaler_list=None, mode='standard'):
    # mode: 'standard' or 'robust'
    n_samples, seq_len, n_channels = data.shape
    data_reshaped = data.reshape(-1, n_channels)  # (n_samples*seq_len, n_channels)
    if scaler_list is None:
        scalers = []
        for ch in range(n_channels):
            if mode == 'standard':
                s = StandardScaler()
            else:
                s = RobustScaler(with_centering=True, with_scaling=True)
            col = data_reshaped[:, ch].reshape(-1, 1)
            s.fit(col)  # 注意：应只在训练域数据上调用此函数进行 fit
            scalers.append(s)
    else:
        scalers = scaler_list

    out = np.empty_like(data_reshaped, dtype=np.float32)
    for ch, s in enumerate(scalers):
        out[:, ch] = s.transform(data_reshaped[:, ch].reshape(-1, 1)).ravel()

    out = out.reshape(n_samples, seq_len, n_channels)
    return out, scalers


def channel_wise_power_then_standardize(data, power_transformers=None):
    n_samples, seq_len, n_channels = data.shape
    resh = data.reshape(-1, n_channels)
    if power_transformers is None:
        ptrans = []
        for ch in range(n_channels):
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            pt.fit(resh[:, ch].reshape(-1,1))  # fit on training domain only
            ptrans.append(pt)
    else:
        ptrans = power_transformers

    transformed = np.empty_like(resh, dtype=np.float32)
    for ch, pt in enumerate(ptrans):
        transformed[:, ch] = pt.transform(resh[:, ch].reshape(-1,1)).ravel()

    # 然后再做 standard scaler 通道独立
    from sklearn.preprocessing import StandardScaler
    scalers = []
    out = np.empty_like(transformed)
    for ch in range(n_channels):
        s = StandardScaler()
        col = transformed[:, ch].reshape(-1,1)
        s.fit(col)  # fit on training domain
        scalers.append(s)
        out[:, ch] = s.transform(col).ravel()

    out = out.reshape(n_samples, seq_len, n_channels)
    return out, ptrans, scalers


class MyDataSet(Dataset):  # 定义类，用于构建数据集
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()
        self.length = label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def data_load_time_series(dataset_path, dataset_label_path, batch_size, target_dataset=False, dataset_name='', classes_number=0):
#     """
#     Args:
#         dataset_path: 数据集路径
#         dataset_label_path: 数据集标签路径
#         batch_size:  每次加载样本数
#         target_dataset: True表示目标域数据集，False表示源域数据集
#         dataset_name:  数据集名称
#         classes_number:  数据集类别数
#         SCARA数据集信号： 6通道电流信号+3通道振动信号
#         PU数据集信号： 两相电流信号+1通道振动信号
#         BJUT_WT数据集信号： 2通道振动信号+1通道编码器信号
#         HUST_motor数据集信号：3振动+1声音
#     Returns:
#     """
#
#     show_figs = False  # True  False
#     data = np.load(dataset_path)
#     if dataset_name == 'SCARA_multi_modal_datasets':  # 原始数据集为6通道电流信号+3通道振动信号
#         # data = data[:, :1024, [0, 1, 2, 3, 4,  5, 6, 7, 8]]
#         data = data[:, :1024, [1, 2, 3, 4, 5, 6, 7, ]]  # 单源域/多源域 五通道电信号（去速度）+xy通道振动
#     else:
#         print('Dataset is not SCARA_multi_modal_datasets, please check the dataset name，len(data)={}! !!!!!!!!!!!!!'.format(len(data)))
#         # data = data[:, :, [0, 1, 2]]  # 北交数据集 通道1：x轴振动信号 通道2：y轴振动信号 通道3：编码器信号；Y轴信号诊断效果最好
#
#     # =============================================== 信号可视化 ======================================== #
#     channels_number = data.shape[2]
#     if show_figs is True:
#         if channels_number <= 3:
#             fig, axes = plt.subplots(channels_number, 1, figsize=(10, 6), sharex=True)  # 第一个为宽，第二个为高
#         else:
#             fig, axes = plt.subplots(channels_number, 1, figsize=(5, 30), sharex=True)
#         for i in range(channels_number):
#             axes[i].plot(data[0:1, :, i].T, linewidth=1)
#             axes[i].set_title(f'Channel {i + 1} Signal')
#             axes[i].set_ylabel('Signal Value')
#         axes[2].set_xlabel('Samples')
#         plt.tight_layout()
#         plt.show()
#     else:
#         pass
#     # =================================================================================================== #
#     # ====================================== 划分训练集、验证集和测试集 ====================================== #
#     label = np.load(dataset_label_path)
#     dataset = MyDataSet(data, label)  # 调用MyDataSet类构建数据集
#     if target_dataset:
#         # 如果是目标域数据集，则不打乱顺序
#         target_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         return target_loader, channels_number
#     else:
#         # 随机选70%、10%、20%做训练集、验证集、测试集
#         train_set_num = int(len(dataset) * 0.7)  # 多源域：70%作为训练集
#         val_set_num = int(len(dataset) * 0.3)  # 多源域：30%作为验证集和测试集
#         test_set_num = len(dataset) - train_set_num - val_set_num
#         train_set, val_set, test_set = random_split(dataset, [train_set_num, val_set_num, test_set_num])
#         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#     # =================================================================================================== #
#         return train_loader, val_loader, test_loader


def data_load_time_series(dataset_path, dataset_label_path, batch_size, target_dataset=False, dataset_name='',
                          classes_number=0, seed=42):
    """
    Args:
        dataset_path: 数据集路径
        dataset_label_path: 数据集标签路径
        batch_size:  每次加载样本数
        target_dataset: True表示目标域数据集，False表示源域数据集
        dataset_name:  数据集名称
        classes_number:  数据集类别数
        SCARA数据集信号： 6通道电流信号+3通道振动信号
        PU数据集信号： 两相电流信号+1通道振动信号
        BJUT_WT数据集信号： 2通道振动信号+1通道编码器信号
        HUST_motor数据集信号：3振动+1声音
        seed: 随机种子
    Returns:
    """
    # 固定随机种子
    set_seed(seed)
    # 构建随机数生成器（确保 random_split 一致）
    g = torch.Generator()
    g.manual_seed(seed)

    show_figs = False  # True  False
    data = np.load(dataset_path)
    if dataset_name == 'SCARA_multi_modal_datasets':  # 原始数据集为6通道电流信号+3通道振动信号
        # data = data[:, :1024, [0, 1, 2, 3, 4,  5, 6, 7, 8]]
        data = data[:, :1024, [1, 2, 3, 4, 5, 6, 7, ]]  # 单源域/多源域 五通道电信号（去速度）+xy通道振动
    else:
        print('Dataset is not SCARA_multi_modal_datasets, please check the dataset name，len(data)={}! !!!!!!!!!!!!!'.format(len(data)))
        # data = data[:, :, [0, 1, 2]]  # 北交数据集 通道1：x轴振动信号 通道2：y轴振动信号 通道3：编码器信号；Y轴信号诊断效果最好

    # =============================================== 信号可视化 ======================================== #
    channels_number = data.shape[2]
    if show_figs is True:
        if channels_number <= 3:
            fig, axes = plt.subplots(channels_number, 1, figsize=(10, 6), sharex=True)  # 第一个为宽，第二个为高
        else:
            fig, axes = plt.subplots(channels_number, 1, figsize=(5, 30), sharex=True)
        for i in range(channels_number):
            axes[i].plot(data[0:1, :, i].T, linewidth=1)
            axes[i].set_title(f'Channel {i + 1} Signal')
            axes[i].set_ylabel('Signal Value')
        axes[2].set_xlabel('Samples')
        plt.tight_layout()
        plt.show()
    else:
        pass
    # =================================================================================================== #
    # ====================================== 划分训练集、验证集和测试集 ====================================== #
    label = np.load(dataset_label_path)
    dataset = MyDataSet(data, label)  # 调用MyDataSet类构建数据集
    if target_dataset:
        # 如果是目标域数据集，则不打乱顺序
        target_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return target_loader, channels_number
    else:
        # 随机选70%、10%、20%做训练集、验证集、测试集
        train_set_num = int(len(dataset) * 0.7)  # 多源域：70%作为训练集
        val_set_num = int(len(dataset) * 0.3)  # 多源域：30%作为验证集和测试集
        test_set_num = len(dataset) - train_set_num - val_set_num
        train_set, val_set, test_set = random_split(dataset, [train_set_num, val_set_num, test_set_num], generator=g)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # =================================================================================================== #
        return train_loader, val_loader, test_loader


# def data_load_time_series1(dataset_path, dataset_label_path, batch_size, target_dataset=False,
#                           dataset_name='', classes_number=0, seed=42):
#     """
#     可复现版本的数据加载函数
#     """
#     # 固定随机种子
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     # 构建随机数生成器（确保 random_split 一致）
#     g = torch.Generator()
#     g.manual_seed(seed)
#
#
#     data = np.load(dataset_path)
#
#     if dataset_name == 'SCARA_multi_modal_datasets':
#         data = data[:, :1024, [1, 2, 3, 4, 5, 6, 7]]
#     else:
#         print('Dataset is not SCARA_multi_modal_datasets, please check dataset name!')
#
#     label = np.load(dataset_label_path)
#     dataset = MyDataSet(data, label)
#     if target_dataset:
#         target_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         channels_number = data.shape[2]
#         return target_loader, channels_number
#     else:
#         train_set_num = int(len(dataset) * 0.7)
#         val_set_num = int(len(dataset) * 0.3)
#         test_set_num = len(dataset) - train_set_num - val_set_num
#
#         # ✅ 使用 generator 控制 random_split 的随机性
#         train_set, val_set, test_set = random_split(dataset,
#                                                     [train_set_num, val_set_num, test_set_num],
#                                                     generator=g)
#
#         # ✅ DataLoader 也使用相同 generator
#         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
#         val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, generator=g)
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, generator=g)
#
#         return train_loader, val_loader, test_loader
