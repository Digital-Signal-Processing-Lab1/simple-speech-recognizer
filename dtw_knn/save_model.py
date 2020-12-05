import pandas as pd
from dm import get_features, distance
import numpy as np
import sklearn.metrics as sm
import sklearn.neighbors as sn
import utils
import scipy.signal as signal
import matplotlib.pyplot as plt
import pickle
import dtw_predictor

# distance = get_dist_func(3)

for win_type in ["hamming"]:
    with open("../dataset/processed/{}.pkl".format(win_type), "rb") as f:
        print(win_type)
        df = pickle.load(f)
        df = df[~df.has_noisy]
        persons_id = list(set(df.person_id))
        predicts = []
        labels = []

        test = persons_id[0]
        contents = set(df.content)
        train = [i for i in persons_id if i != test]
        train_data = df.apply(lambda d: d.person_id != test, axis=1)
        test_data = df.apply(lambda d: d.person_id == test, axis=1)
        padded_data, max_length = get_features(df.wave_data, 20, window=signal.hanning(512))
        with open("./data.pkl", "wb") as f:
            data = {'data':padded_data, 'max_length':max_length, 'label':df.content}
            pickle.dump(data, f)
        model = sn.KNeighborsClassifier(n_neighbors=3, metric=distance, metric_params={'k':60})
        model.fit(X=padded_data[train_data], y=df[train_data].content)
        # predictor = dtw_predictor.Predictor(model, max_length, predict_win=win_type)
        print("train_predict: {:.4f}".format(sm.accuracy_score(df[train_data].content, model.predict(padded_data[train_data]))))
