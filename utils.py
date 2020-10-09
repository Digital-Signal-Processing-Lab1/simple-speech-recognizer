import pickle
import numpy as np
import pandas as pd


def load_pkl(path):
    with open(path, "rb") as pkl:
        df = pickle.load(pkl)
        return df


def padding_zeros_to(data: pd.Series, length):
    ret = data.copy()
    for i in range(len(ret)):
        t = np.array(data[i]).reshape(-1)
        ret[i] = np.concatenate([t, np.zeros(length-t.shape[0])], axis=0)
    return ret


def padding_to_max(data: pd.Series):
    max_length = np.max([len(w) for w in data])
    ret = padding_zeros_to(data, max_length)
    return ret, max_length


def resample(source_signal: np.ndarray, source_fs: int, target_fs: int):
    source = source_signal.flatten()
    signal_length = source.shape[0]
    signal_endure = (signal_length - 1) / source_fs
    tp = np.linspace(0, signal_endure, signal_length)
    target_length = int(signal_endure * target_fs)
    t = np.linspace(0, signal_endure, target_length)
    target = np.interp(t, tp, source)
    target_signal = target.reshape([-1, 1])
    return target_signal


