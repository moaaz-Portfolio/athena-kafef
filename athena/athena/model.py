"""Four-class CNN + BiLSTM + attention activity classifier."""
import torch
import torch.nn as nn


class ActivityNet(nn.Module):
    def __init__(self, n_feat=64, n_cls=4, drop=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop * 0.3),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(drop * 0.3),
        )
        self.temporal = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_cls))

    def forward(self, x):                       # x: (batch, time, features)
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.temporal(h)
        a = torch.softmax(self.attention(h), dim=1)
        return self.head((a * h).sum(dim=1))

