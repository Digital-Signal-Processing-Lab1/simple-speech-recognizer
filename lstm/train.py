from lstm.model import Model
import sklearn.metrics as sm
import torch
import utils
import numpy as np
from librosa.feature import mfcc

epoch = 500
data_dim = 1


def reader(data, label, batch_size, shuffle=True):
    n = data.shape[0]
    index = np.arange(n)
    if shuffle:
        np.random.shuffle(index)
    m = int(n/batch_size)
    def _reader():
        for i in range(m):
            yield_label = label[index[i:min(n, (i+1)*batch_size)]]
            yield_data = data[index[i:min(n, (i+1)*batch_size)]]
            yield yield_data, yield_label

    return _reader


def feeder(d):
    tmp = torch.tensor(d, dtype=torch.float)
    tmp = torch.transpose(tmp, 1, 2)
    tmp = torch.transpose(tmp, 0, 1)
    return tmp


data = utils.load_pkl("../dataset/processed/rect.pkl")
persons_id = np.unique(data.person_id)
test_id = persons_id[0]
train_id = [i for i in persons_id if i != test_id]
train_index = data.apply(lambda d: d.person_id != test_id, axis=1)
test_index = data.apply(lambda d: d.person_id == test_id, axis=1)
wave_data = data.wave_data
label = torch.zeros([len(wave_data), 10])
for i in range(len(wave_data)):
    label[i, int(data.content[i])] = 1
wave_data = wave_data.apply(lambda d: (d-np.mean(d))/(np.std(d)))
wave_data, max_length = utils.padding_to_max(wave_data)
features = []
for x in wave_data:
    t1 = mfcc(x, n_mfcc=20, sr=16000, n_fft=512, hop_length=256)
    t2 = utils.diff(t1, delta=1)
    t3 = utils.diff(t1, delta=2)
    t = np.concatenate([t1, t2, t3], axis=0)
    features.append(t)
features = np.array(features)
model = Model()
model.reader = reader(features[train_index], label[train_index], 64)
model.feeder = feeder
model.test_label = data.content[test_index]
model.test_data = features[test_index]
model.create_model(60, 60, 1)
model.set_optim(0.01)
try:
    model.train(epoch)
    model.test()
    model.save_model("./model/model2/")
except KeyboardInterrupt:
    model.save_model("./model/model2/")


