import scipy.io as scio
import numpy as np
import os
import torch
import torch.nn.functional
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.model_selection import train_test_split
import random


def get_labels(label_path):
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def label_2_onehot(label_list):
    look_up_table = {-1: [1, 0, 0],
                     0: [0, 1, 0],
                     1: [0, 0, 1]}
    label_onehot = [np.asarray(look_up_table[label]) for label in label_list]
    return label_onehot


def norm_fun(m, type=0):
    if type == 0:  # 0,1平均归一化
        mmax = m.max()
        mmin = m.min()
        m = (m - mmin) / (mmax - mmin)
    elif type == 1:  # Z-score归一化
        mavg = m.mean()
        mstd = m.std()
        m = (m - mavg) / (mstd)
    return m


def Data_2D_Interpolator(a):
    a = np.ma.masked_invalid(a)
    x = np.arange(0, a.shape[1])
    y = np.arange(0, a.shape[0])
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~a.mask]
    y1 = yy[~a.mask]
    newarr = a[~a.mask].data

    # 二维差值使用CloughTocher2DInterpolator
    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic', fill_value=-20)

    return GD1


# 2D化处理
def build_2D_DE_data(raw_DE_data, map_size=8, freq_num=5):
    locs = pd.read_csv("./data_set/loca62_8x9_test.csv")
    locs_data = np.asarray(locs)
    locs_xy = np.stack((locs_data[:, 5], locs_data[:, 4], locs_data[:, 3]), axis=1)
    map_size1 = map_size + 1

    feature_sample = np.zeros((0, map_size, map_size1))

    for freq in range(np.size(raw_DE_data, 0)):
        feature_map = np.full((map_size, map_size1), 0.00)
        label_name = np.full((map_size, map_size1), str(0.00))
        for c in range(np.size(raw_DE_data[freq], 0)):
            feature_map[locs_xy[c][0]][[locs_xy[c][1]]] = raw_DE_data[freq][c]
            label_name[locs_xy[c][0]][[locs_xy[c][1]]] = locs_xy[c][2]
            feature_map = Data_2D_Interpolator(feature_map)
        feature_sample = np.append(feature_sample, np.expand_dims(feature_map, axis=0), axis=0)
        del feature_map

    return feature_sample


# 处理生成DE二维数据dict
# dis
def build_DE_eeg_dataset(folder_path, origin_path, dis=6, map_size=8, subject="1"):
    feature_vector_dict = {}  # 总特征向量字典
    label_dict = {}  # 总标签字典
    if (os.path.exists(folder_path + 'feature_2D.npy')):
        if (os.path.exists(folder_path + 'label_2D.npy')):
            print("---数据已存在---")
            # -----注意读大型npy字典需要np.load().item()恢复字典-----
            feature_vector_dict = np.load(folder_path + 'feature_2D.npy', allow_pickle=True).item()
            label_dict = np.load(folder_path + 'label_2D.npy', allow_pickle=True).item()
            return feature_vector_dict, label_dict

    # 需保存处理数据集——>npy
    labels = get_labels(os.path.join(origin_path, 'label.mat'))

    try:
        all_mat_file = os.walk(origin_path)
        skip_set = {'label.mat'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:

                if file_name not in skip_set:
                    file_cnt += 1
                    print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))

                    all_trials_dict = scio.loadmat(os.path.join(origin_path, file_name),
                                                   verify_compressed_data_integrity=False)
                    experiment_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}

                    for key in all_trials_dict.keys():  # keys，trail的数据矩阵名列表
                        if 'eeg' not in key:
                            continue
                        feature_vector_list = []
                        label_list = []
                        cur_trial = np.asarray(all_trials_dict[key])  # 当前trail
                        length = len(cur_trial[1])
                        cur_trial = np.transpose(cur_trial, (1, 0, 2))
                        pos = 0
                        while pos + dis <= length:
                            feature_sample = []
                            raw_data = np.transpose(np.asarray(cur_trial[pos: pos + dis]), (0, 2, 1))

                            for x in raw_data:
                                feature_sample.append(build_2D_DE_data(x))
                            feature_vector_list.append(np.asarray(feature_sample))
                            raw_label = labels[int(key.split('_')[-2][3:]) - 1]
                            label_list.append(raw_label)

                            del feature_sample
                            del raw_data
                            pos += dis
                        trial = key.split('_')[1][3:]
                        feature_vector_trial_dict[trial] = np.asarray(feature_vector_list)
                        label_trial_dict[trial] = np.asarray(label_2_onehot(label_list))

                    feature_vector_dict[experiment_name] = feature_vector_trial_dict
                    label_dict[experiment_name] = label_trial_dict
                else:
                    continue

    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    np.save(folder_path + 'feature_2D.npy', feature_vector_dict, allow_pickle=True)
    np.save(folder_path + 'label_2D.npy', label_dict, allow_pickle=True)

    return feature_vector_dict, label_dict


# 划分数据集方式1
# 跨被试划分
def subject_cross_data_split(feature_vector_dict, label_dict, test_subject_set, dataset):
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []

    if dataset == 'Cognitive':
        for subject in feature_vector_dict.keys():  # each subject
            for trial in feature_vector_dict[subject].keys():  # all trails
                if str(trial) in test_subject_set:
                    test_feature.extend(feature_vector_dict[subject][trial])
                    test_label.extend(label_dict[subject][trial])
                else:
                    train_feature.extend(feature_vector_dict[subject][trial])
                    train_label.extend(label_dict[subject][trial])

    else:
        for subject in feature_vector_dict.keys(): # each subject
            for trial in feature_vector_dict[subject].keys(): # all trails
                if subject in test_subject_set:
                    test_feature.extend(feature_vector_dict[subject][trial])
                    test_label.extend(label_dict[subject][trial])
                else:
                    train_feature.extend(feature_vector_dict[subject][trial])
                    train_label.extend(label_dict[subject][trial])
    print("Partition1-cross-subject finished!")
    return train_feature, train_label, test_feature, test_label

# 划分数据集方式2
# 无跨划分
def avg_cross_data_split(feature_vector_dict, label_dict, dataset):
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []

    for experiment in feature_vector_dict.keys():
        for trial in feature_vector_dict[experiment].keys():
            X_train, X_test, y_train, y_test = train_test_split(feature_vector_dict[experiment][trial], label_dict[experiment][trial], test_size=0.2, random_state=3, stratify=None)
            test_feature.extend(X_test)
            test_label.extend(y_test)
            train_feature.extend(X_train)
            train_label.extend(y_train)
    print("Partition2 finished!")
    return train_feature, train_label, test_feature, test_label

# 无跨划分
def data_split(feature_vector_dict, label_dict):
    train_feature = []
    train_label = []

    for experiment in feature_vector_dict.keys():
        for trial in feature_vector_dict[experiment].keys():
            train_feature.extend(feature_vector_dict[experiment][trial])
            train_label.extend(label_dict[experiment][trial])

    # 随机打乱
    index = [i for i in range(len(train_feature))]
    random.shuffle(index)
    train_feature = torch.from_numpy(np.array(train_feature))[index]
    train_label = torch.from_numpy(np.array(train_label))[index]
    return train_feature, train_label


# dataSet继承定义
class EEGDataset_DE(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list):
        self.feature_list = feature_list
        self.label_list = label_list

    def __getitem__(self, index):
        feature = torch.from_numpy(self.feature_list[index])
        label = torch.from_numpy(self.label_list[index]).long()
        label = torch.argmax(label)
        return feature, label

    def __len__(self):
        return len(self.label_list)


# dataSet继承定义
class EEGDataset_DE_tensor(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list):
        self.feature_list = feature_list
        self.label_list = label_list

    def __getitem__(self, index):
        feature = self.feature_list[index]
        label = self.label_list[index].long()
        label = torch.argmax(label)
        return feature, label

    def __len__(self):
        return len(self.label_list)
