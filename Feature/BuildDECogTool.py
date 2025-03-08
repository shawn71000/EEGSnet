import scipy.io as scio
import numpy as np
import os
import torch
import torch.nn.functional
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import StandardScaler


def get_labels(label_path):
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def label_2_onehot(label_list):
    look_up_table = {0: [1, 0, 0],
                     1: [0, 1, 0],
                     2: [0, 0, 1]}
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

# 2D化处理5*8
def build_2D_DE_data(raw_DE_data, map_size=7, freq_num=5):
    locs = pd.read_csv("./data_set/loca32_5x5_test.csv")
    locs_data = np.asarray(locs)
    locs_xy = np.stack((locs_data[:, 5], locs_data[:, 4], locs_data[:, 3]), axis=1)
    map_size1 = map_size - 2

    feature_sample = np.zeros((0, map_size, map_size1))

    for freq in range(np.size(raw_DE_data, 0)):
        feature_map = np.full((map_size, map_size1), 0.00)
        label_name = np.full((map_size, map_size1), str(0.00))
        for c in range(np.size(raw_DE_data[freq], 0)):
            feature_map[locs_xy[c][0]][[locs_xy[c][1]]] = raw_DE_data[freq][c]
            label_name[locs_xy[c][0]][[locs_xy[c][1]]] = locs_xy[c][2]
            feature_map = Data_2D_Interpolator(feature_map)
            # feature_map = norm_fun(feature_map)
        feature_sample = np.append(feature_sample, np.expand_dims(feature_map, axis=0), axis=0)
        del feature_map

    return feature_sample


# 2D化处理9*9
def build_2D_DE_data_99(raw_DE_data, map_size=7, freq_num=5):
    locs = pd.read_csv("./data_set/loca32_9x9.csv")
    locs_data = np.asarray(locs)
    locs_xy = np.stack((locs_data[:, 5], locs_data[:, 4], locs_data[:, 6]), axis=1)

    feature_sample = np.zeros((0, map_size, map_size))

    for freq in range(np.size(raw_DE_data, 0)):
        feature_map = np.full((map_size, map_size), 0.00)
        i = 0
        for c in range(np.size(raw_DE_data[freq], 0)):
            locs_32 = locs_xy[:, 2]
            while locs_32[i] == c + 1:
                feature_map[locs_xy[i][0]][[locs_xy[i][1]]] = raw_DE_data[freq][c]
                i += 1
            else:
                i += 1

            # std = StandardScaler()
            # feature_map = std.fit_transform(feature_map)
            feature_map = Data_2D_Interpolator(feature_map)
        feature_sample = np.append(feature_sample, np.expand_dims(feature_map, axis=0), axis=0)
        del feature_map

    return feature_sample


# 处理生成DE二维数据dict
# dis
def build_DE_eeg_dataset(folder_path, origin_path, dis=6, map_size=7, subject="1"):
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
            for file_name in dir_list:
                if file_name == folder_path.split('_')[-1].split('/')[0]:
                    for sub_path, sub_file_name, sub_file_list in os.walk(os.path.join(origin_path, file_name)):
                        file_cnt += 1
                        print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(dir_list)))
                        cog_num = 0
                        for sub_cog in sub_file_name:
                            Cog_mat = os.walk(os.path.join(sub_path, sub_cog))
                            for mat_file_path, mat_list, mat_file_list in Cog_mat:
                                feature_vector_trial_dict = {}
                                label_trial_dict = {}
                                for m, mat_file in enumerate(mat_file_list):
                                    all_trials_dict = scio.loadmat(os.path.join(mat_file_path, mat_file), verify_compressed_data_integrity=False)
                                    feature_vector_list = []
                                    label_list = []
                                    cur_trial = np.asarray(all_trials_dict['de_feature'])  # Num * channel * frequency
                                    length = cur_trial.shape[0]
                                    # cur_trial = np.transpose(cur_trial, (1, 0, 2))
                                    pos = 0
                                    while pos + dis <= length:
                                        feature_sample = []
                                        raw_data = np.transpose(np.asarray(cur_trial[pos: pos + dis]), (0, 2, 1))

                                        for x in raw_data:
                                            feature_sample.append(build_2D_DE_data(x))
                                        feature_vector_list.append(np.asarray(feature_sample))
                                        raw_label = labels[cog_num]
                                        label_list.append(raw_label)

                                        del feature_sample
                                        del raw_data
                                        pos += dis
                                    feature_vector_trial_dict[m] = np.asarray(feature_vector_list)
                                    label_trial_dict[m] = np.asarray(label_2_onehot(label_list))

                                feature_vector_dict[sub_cog] = feature_vector_trial_dict
                                label_dict[sub_cog] = label_trial_dict
                            cog_num += 1

                    np.save(folder_path + 'feature_2D.npy', feature_vector_dict, allow_pickle=True)
                    np.save(folder_path + 'label_2D.npy', label_dict, allow_pickle=True)

    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

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
def avg_cross_data_split(feature_vector_dict, label_dict):
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []

    for experiment in feature_vector_dict.keys():
        cnt = 0
        for trial in feature_vector_dict[experiment].keys():
            if cnt % 5 == 0:
                test_feature.extend(feature_vector_dict[experiment][trial])
                test_label.extend(label_dict[experiment][trial])
            else:
                train_feature.extend(feature_vector_dict[experiment][trial])
                train_label.extend(label_dict[experiment][trial])
            cnt += 1
    print("Partition2 finished!")
    return train_feature, train_label, test_feature, test_label


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
