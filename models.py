# @Author: yusuf
# @Date:   2019-12-10T11:51:06+02:00
# @Last modified by:   yusuf
# @Last modified time: 2019-12-18T08:59:21+02:00


import torch.nn as nn
import torch

class DeepFake(nn.Module):
    def __init__(self):
        super(DeepFake, self).__init__()
        self.LSTM1 = nn.LSTM(2048, 2048, 2, dropout=0.5, batch_first=True)
        self.LSTM2 = nn.LSTM(2048, 512, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(562, 248)
        self.fc2 = nn.Linear(248, 2)
        
    def forward(self, x, y):
        x, _hidden = self.LSTM1(x.float())
        x = self.relu(x)
        h = _hidden[0].mean(0).unsqueeze(0)
        c = _hidden[1].mean(0).unsqueeze(0)
        x, _ = self.LSTM2(x) #, (h, c))
        combined = torch.cat((x[:, -1, :], y.squeeze(1)), 1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x

def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
