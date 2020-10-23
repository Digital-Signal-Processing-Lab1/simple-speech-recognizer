import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pkl(path) -> pd.DataFrame:
    with open(path, "rb") as pkl:
        df = pickle.load(pkl)
        return df


def padding_zeros_to(data: pd.Series, length) -> np.array:
    ret = np.zeros([len(data), length], dtype=float)
    for i in range(len(data)):
        t = np.array(data.iat[i]).reshape(-1)
        ret[i, :t.shape[0]] = t
    return ret


def padding_to_max(data: pd.Series) -> (np.ndarray, int):
    max_length = np.max([len(w) for w in data])
    ret = padding_zeros_to(data, max_length)
    return ret, max_length


def resample(source_signal: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    source = source_signal.flatten()
    signal_length = source.shape[0]
    signal_endure = (signal_length - 1) / source_fs
    tp = np.linspace(0, signal_endure, signal_length)
    target_length = int(signal_endure * target_fs)
    t = np.linspace(0, signal_endure, target_length)
    target = np.interp(t, tp, source)
    target_signal = target.reshape([-1, 1])
    return target_signal


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
