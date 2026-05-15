import pandas as pd


def train_history():
    train_hist = {}

    train_hist['src_train_time_loss'] = []
    train_hist['src_train_time_freq_loss'] = []
    train_hist['src_train_freq_loss'] = []
    train_hist['src_train_total_loss'] = []
    train_hist['src_test_acc'] = []
    train_hist['src_test_loss'] = []
    train_hist['src_val_acc'] = []
    train_hist['src_val_loss'] = []
    train_hist['tar_test_acc'] = []
    train_hist['tar_test_loss'] = []
    train_hist['val_acquire_best_acc'] = []

    return train_hist


def DDC_train_history():
    train_hist = {}

    train_hist['src_and_tgt_mmd_loss'] = []
    train_hist['yhmmd_loss'] = []
    train_hist['total_loss'] = []
    train_hist['Source_test_acc'] = []
    train_hist['Source_test_loss'] = []
    train_hist['Target_test_acc'] = []
    train_hist['Target_test_loss'] = []

    return train_hist


def save_his(train_hist={}, save_name=''):
    """
    save history data
    """
    data_df = pd.DataFrame(train_hist)
    data_df.to_csv(save_name + '.csv')


def save_predict_labels(yt_label=None, yt_pre_label=None, save_name=''):
    """
    save best predict labels
    :param yt_label: target labels
    :param yt_pre_label: target predicted labels
    :param save_name: save file name
    :return:
    """
    best_prediction_labels = {}
    best_prediction_labels['yt_pre_label'] = []
    best_prediction_labels['yt_label'] = []
    yt_label = yt_label.cpu().detach().data.numpy()
    yt_pre_label_new = yt_pre_label.cpu().detach().data.numpy()
    best_prediction_labels['yt_label'] = yt_label
    best_prediction_labels['yt_pre_label'] = yt_pre_label_new
    prediction_lab = pd.DataFrame(best_prediction_labels)
    prediction_lab.to_csv(save_name)


