import utils
import numpy as np
import pickle
from librosa.feature import mfcc
from dtw_knn.dm import distance, normalize
from sklearn.neighbors import KNeighborsClassifier


def get_features(wave_data: np.ndarray, n_mfcc, window):
    """
    feature
    """
    x = wave_data
    features = []
    for i in range(x.shape[0]):
        t1 = mfcc(x[i], sr=16000, n_mfcc=n_mfcc, n_fft=512, hop_length=256, window=window)
        t2 = utils.diff(t1, axis=0)
        t3 = utils.diff(t1, axis=0, delta=2)
        t = np.concatenate([t1, t2, t3], axis=0).flatten()
        features.append(t)
    return np.array(features)


class Predictor:
    def __init__(self, data_path, n_mfcc=20, fs=16000, detect_win=1, predict_win=1):
        self.model = KNeighborsClassifier(n_neighbors=3, metric=distance, metric_params={'k':3*n_mfcc})
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.model.fit(data['data'], data['label'])
        self.max_length = data['max_length']
        self.n_mfcc = n_mfcc
        self.target_fs = fs
        self.detect_win = detect_win
        self.predict_win = predict_win

    def predict(self, data):
        _data = normalize(data).reshape(-1)
        _data = np.pad(_data, (0, self.max_length - len(_data))).reshape([1, -1])
        features = get_features(_data, n_mfcc=self.n_mfcc, window=self.predict_win)
        _predict = self.model.predict(features)
        return _predict
