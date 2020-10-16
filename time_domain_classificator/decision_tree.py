import pickle
import pandas as pd
import numpy as np
import sklearn.tree as st
import sklearn.metrics as sm
import utils


def get_features(wave_data: pd.Series):
    """
    feature: abs, mean, std, diff_mean, diff_std
    """
    feature_dim = 9
    n = len(wave_data)
    features = np.empty([n, feature_dim])
    for i, d in enumerate(wave_data):
        diff_d = np.diff(d)
        diff_diff_d = np.diff(diff_d)
        features[i] = [
            np.sum(np.abs(d)),
            np.mean(d),
            np.var(d),
            np.sum(np.abs(diff_d)),
            np.mean(diff_d),
            np.var(diff_d),
            np.sum(np.abs(diff_diff_d)),
            np.mean(diff_diff_d),
            np.var(diff_diff_d),
        ]
    return features


test = [1, 2]
with open("../dataset/processed/rect.pkl", "rb") as f:
    df = pickle.load(f)
    df = df[~df.has_noisy]
    persons_id = set(df.person_id)
    print(persons_id)
    contents = set(df.content)
    train = [i for i in persons_id if i not in test]
    train_data = df[df.apply(lambda d: d.person_id not in test, axis=1)]
    test_data = df[df.apply(lambda d: d.person_id in test, axis=1)]
    features = get_features(train_data.wave_data)
    model = st.DecisionTreeClassifier()
    model.fit(features, train_data.content)
    print(model.feature_importances_)
    features = get_features(test_data.wave_data)
    predict = model.predict(features)
    print(test_data.content.to_numpy())
    print(predict)
    r2 = sm.r2_score(test_data.content, predict)
    print(sm.accuracy_score(test_data.content, predict))
    print(r2)
    print(sm.f1_score(test_data.content.to_numpy(),predict, average=None))
    utils.plot_classify_result(range(10), test_data.content, predict, "result.png")
