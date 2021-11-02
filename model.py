import torch
from torch import nn


class STTModel(nn.Module):
    def __init__(self, rnn_input_size=128, hidden_size=256):
        super(STTModel, self).__init__()

        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
                nn.Conv1d(81,81,10,2,padding=10//2),
                nn.BatchNorm1d(81),
                nn.GELU(),
                nn.Dropout(.1))

        self.dense = nn.Sequential(
                nn.Linear(81, rnn_input_size),
                nn.LayerNorm(rnn_input_size),
                nn.GELU(),
                nn.Dropout(.1),
                nn.Linear(rnn_input_size, rnn_input_size),
                nn.LayerNorm(rnn_input_size),
                nn.GELU(),
                nn.Dropout(.1),
                )

        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size,
                num_layers=1, dropout=0.0, bidirectional=True)

        self.fc = nn.Sequential(
                nn.LayerNorm(hidden_size*2),
                nn.Dropout(.1),
                nn.Linear(hidden_size*2, 28))

    def forward(self, x, hidden):
        x = x.squeeze(1)
        x = self.conv(x)
        x = x.transpose(1,2)
        x = self.dense(x)
        x = x.transpose(0,1)
        x, (hn, cn)  = self.lstm(x, hidden)
        x = self.fc(x)
        return x, (hn, cn)

    @staticmethod
    def get_initial_hidden(batch_size):
        n, hs = 1, 256
        return (torch.zeros(n*2, batch_size, hs),
                torch.zeros(n*2, batch_size, hs))
