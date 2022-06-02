import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(data):
    data_dir = os.path.join(data)
    data = pd.read_csv(data_dir,parse_dates=['date'])
    data.index = data['date']
    data = data.drop('date', axis=1)
    return data


def split_sequence_uni_step(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], [sequence[end_ix][0]]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_sequence_multi_step(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)[:, :, 0]


def data_loader(x, y, train_split, test_split, batch_size):
    train_seq, test_seq, train_label, test_label = train_test_split(x, y, train_size=train_split, shuffle=False)
    val_seq, test_seq, val_label, test_label = train_test_split(test_seq, test_label, train_size=test_split,
                                                                shuffle=False)
    train_set = TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_label))
    val_set = TensorDataset(torch.from_numpy(val_seq), torch.from_numpy(val_label))
    test_set = TensorDataset(torch.from_numpy(test_seq), torch.from_numpy(test_label))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_lr_scheduler(lr_scheduler, optimizer):
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01,
                                                               patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape


def plot_pred_test(pred, actual, path, feature, model, step):
    plt.figure(figsize=(10, 8))

    plt.plot(pred, label='Pred')
    plt.plot(actual, label='Actual')

    plt.xlabel('Time/Hour', fontsize=18)
    plt.ylabel('tplb'.format(feature), fontsize=18)

    plt.legend(loc='best')
    plt.grid()

    if model.__class__.__name__ == 'AttentionalLSTM':
        model.__class__.__name__ = 'AB-LSTM'
    plt.title('Prediction using {}'.format(model.__class__.__name__), fontsize=18)
    plt.savefig(os.path.join(path, '{} Prediction using {} and {}.png'.format(feature, model.__class__.__name__, step)))


def inverse_transform_col(scaler, y, n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y
