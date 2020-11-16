import utils
import numpy as np
import pickle
from librosa.feature import mfcc

class Predictor:
    def __init__(self, model, max_length):
        self.model = model
        self.max_length = max_length
        self.n_mfcc = 20
        self.target_fs = 16000

    def predict(self, data):
        _data = (data - np.mean(data)) / np.std(data)
        if self.max_length < _data.shape[0]:
            _data = _data[:self.max_length]
        else:
            _data = np.pad(_data, (0, self.max_length - data.shape[0]))
        t1 = mfcc(_data, sr=16000, n_mfcc=self.n_mfcc)
        t2 = utils.diff(t1, axis=0)
        t3 = utils.diff(t1, axis=0, delta=2)
        t = np.concatenate([t1.T, t2.T, t3.T], axis=1).flatten()
        features = np.array([t])
        _predict = self.model.predict(features)
        return _predict


def save_predictor(predictor: Predictor, filename):
    f = open(filename, 'wb')
    pickle.dump(predictor, f)


def load_predictor(filename) -> Predictor:
    f = open(filename, 'rb')
    return pickle.load(f)