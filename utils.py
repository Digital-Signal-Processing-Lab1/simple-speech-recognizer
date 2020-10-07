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
