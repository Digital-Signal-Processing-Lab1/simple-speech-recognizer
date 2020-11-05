# -*- coding:utf-8 -*-

import sys
# 注意在本机需要添加系统默认路径
sys.path.append(r"C:\Users\lenovo\simple-speech-recognizer")
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import sklearn.utils as su
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.metrics as sm
import scipy.signal as signal
import os
import VAD
import utils
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
N, M = 256, 128
file_path = "../dataset/processed/"

def raw_feature_pre(store, energy, amplitude, zerocrossingrate, label):
    d = []
    d.append(energy)
    d.append(amplitude)
    d.append(zerocrossingrate)
    d.append(label)
    store.append(d)

def padding_to_max(data: pd.Series):
    max_length = np.max([len(w) for w in data.energy])
    data.energy = padding_zeros_to(data.energy, max_length)
    data.amplitude = padding_zeros_to(data.amplitude, max_length)
    data.zerocrossingrate = padding_zeros_to(data.zerocrossingrate, max_length)
    return data, max_length

def padding_zeros_to(data: pd.Series, length):
    ret = data.copy()
    for i in range(len(ret)):
        t = np.array(data[i]).reshape(-1)
        ret[i] = np.concatenate([t, np.zeros(length-t.shape[0])], axis=0)
    return ret


def downsampling(array, dim):
    array_len = len(array)
    if array_len >= dim:
        inc = int(array_len/dim)
        ret = array[:array_len:inc]
        return ret
    else : return -1

def plot_classify_result(label, real, predict, filename):
    plt.figure(figsize=(8, 6))
    n_label = len(label)
    m = np.zeros([n_label, n_label])
    for r, p in zip(real, predict):
        m[int(r), int(p)] += 1
    plt.imshow(m)
    plt.xticks(label)
    plt.yticks(label)
    plt.xlabel("predict")
    plt.ylabel("real")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)


for window_type in ["rect", "hamming", "hanning"]:
    if window_type == 'rect':
        winfunc = 1
    elif window_type == 'hamming':
        winfunc = signal.windows.hamming(N)
    else:
        winfunc = signal.windows.hann(N)
    print("training with window of {}...".format(window_type))

    path = os.path.join(file_path, "{}.pkl".format(window_type))
    df = utils.load_pkl(path)
    # 去除噪声数据集
    # df = df[df["has_noisy"] == False]
    # df = df.reset_index()
    wave_data = df.wave_data
    label = df.content
    store = []

    # 减采样降维
    for i in range(len(wave_data)):
        frames, num_frame = VAD.addWindow(wave_data[i].reshape(-1, 1), N, M, winfunc)
        energy = VAD.calEnergy(frames, N).reshape(1, num_frame).flatten()
        energy = downsampling(energy, 33)
        amplitude = VAD.calAmplitude(frames, N).reshape(1, num_frame).flatten()
        amplitude = downsampling(amplitude, 33)
        zerocrossingrate = VAD.calZeroCrossingRate(frames, N).reshape(1, num_frame).flatten()
        zerocrossingrate = downsampling(zerocrossingrate, 33)
        raw_feature_pre(store, energy, amplitude, zerocrossingrate, label[i])
    raw_feature_df = pd.DataFrame(store, columns=["energy", "amplitude", "zerocrossingrate", "label"])
    raw_feature_df, max_length = padding_to_max(raw_feature_df)

    # 开始降维
    # time_domain_f 为特征向量
    energy_ss, amplitude_ss ,zerocrossingrate_ss = [], [], []
    for i in range(len(raw_feature_df)):
        energy_ss.append(raw_feature_df["energy"][i])
        amplitude_ss.append(raw_feature_df["amplitude"][i])
        zerocrossingrate_ss.append(raw_feature_df["zerocrossingrate"][i])
    energy_ss = np.array(energy_ss).reshape(-1, max_length)
    amplitude_ss = np.array(amplitude_ss).reshape(-1, max_length)
    zerocrossingrate_ss = np.array(zerocrossingrate_ss).reshape(-1, max_length)
    time_domain_f = np.concatenate([energy_ss, amplitude_ss, zerocrossingrate_ss], axis=1)
    time_domain_f = time_domain_f / np.max(time_domain_f)
# 使用KNN分类器识别语音信号，
    x, y = su.shuffle(time_domain_f, np.array(raw_feature_df["label"]).astype(np.int).reshape(-1, 1), random_state=40)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=40, train_size=0.66, test_size=0.34)
    #knn = KNeighborsClassifier(n_neighbors=3)
    #knn.fit(X_train,y_train.ravel())

#使用10折交叉验证,求出使得KNN算法性能最好的K值
    #scores = cross_val_score(knn, X_train,y_train.ravel(), cv=10, scoring='accuracy')
    k_range = range(1,31)
    k_scores = []
    for k in k_range:
        knn_t = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_t, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print("cross-validation average scores for different 'k':",k_scores)
    best_k = 1
    best_score = 0
    for m in range(29):
        if k_scores[m] > best_score:
            best_score = k_scores[m]
            best_k = m+1
    print("best 'k' for knn:",best_k)
#KNN算法采用用计算所得的最佳K值
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    #knn_best = KNeighborsClassifier(n_neighbors=3)
    knn_best.fit(X_train,y_train.ravel())
    print("train accuracy :",knn_best.score(X_train,y_train))
    print("test  accuracy :",knn_best.score(X_test,y_test))
    print("best cross-validation average score:",best_score)

    test = y_test.reshape(1, -1).flatten()
    predict = knn_best.predict(X_test)
    r2 = sm.r2_score(test, predict)
    print("r2 score :", r2)
    print("f1 score :", sm.f1_score(test, predict, average=None))
    print("plotting data...")
    plot_classify_result(range(10), test, predict, "../time_domain_classificator/KNN_result_"+window_type+".png")

    print("done")

