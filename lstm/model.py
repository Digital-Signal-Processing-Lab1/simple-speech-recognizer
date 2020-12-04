import torch
import torch.nn as nn
import torch.optim as optim
import os
import sklearn.metrics as sm


class Classify(nn.Module):
    def __init__(self, input_dim, hidden_size, class_num, num_layers):
        super(Classify, self).__init__()
        self.pre_fc = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.sel_fc = nn.Linear(2, 1)
        self.post_fc = nn.Linear(hidden_size, class_num, bias=False)
        self.hidden_size=hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        tmp = self.pre_fc(x)
        tmp, (h, c) = self.lstm(tmp)
        seq_l, batch, _ = x.shape
        tmp = h.view(self.num_layers, 2, batch, self.hidden_size)
        tmp = torch.transpose(tmp, 1, 3)
        tmp = self.sel_fc(tmp)
        tmp = torch.squeeze(tmp, 3)
        tmp = torch.squeeze(tmp, 0)
        tmp = torch.transpose(tmp, 0, 1)
        tmp = self.post_fc(tmp)
        tmp = torch.sigmoid(tmp)
        return tmp


class Model:
    def __init__(self):
        self.classifier = None
        self.reader = None
        self.feeder = None
        self.input = None
        self.optimizer = None
        self.error = None
        self.test_data = None
        self.test_label = None
        self.test_lengths = None
        self.loss = nn.BCELoss()

    def train(self, epoch):
        for i in range(epoch):
            self.train_one_epoch()
            print(i, self.error)
            if i%10 == 0:
                self.test()

    def test(self):
        prob = self.predict(self.test_data)
        prob = torch.squeeze(prob, 0)
        predict = torch.argmax(prob, dim=1)
        print("acc=", sm.accuracy_score(self.test_label, predict))

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.classifier, os.path.join(path, "classifier.pkl"))

    def train_one_epoch(self):
        for data, label in self.reader():
            self.input = self.feeder(data)
            prob = self.classifier(self.input)
            prob = torch.squeeze(prob, 0)
            self.error = self.loss(prob, label)
            self.error.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def set_optim(self, lr):
        self.optimizer = optim.Adam(lr=lr, params=self.classifier.parameters())

    def load_model(self, path):
        self.classifier = torch.load(os.path.join(path, "classifier.pkl"))

    def create_model(self, in_c, hidden_dim, num_layers):
        self.classifier = Classify(in_c, hidden_dim, 10, num_layers)

    def predict(self, x):
        with torch.no_grad():
            x = self.feeder(x)
            prob = self.classifier(x)
        return prob