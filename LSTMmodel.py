import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class PredictorLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(PredictorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_line0 = torch.nn.LSTM(input_size=input_size[0], hidden_size=hidden_size[0],
                                        num_layers=num_layers, batch_first=True)

        self.lstm_line1 = torch.nn.LSTM(input_size=input_size[1], hidden_size=hidden_size[1],
                                       num_layers=num_layers, batch_first=True)

        self.lstm_line2 = torch.nn.LSTM(input_size=input_size[2], hidden_size=hidden_size[2],
                                       num_layers=num_layers, batch_first=True)

        self.lstm_line3 = torch.nn.LSTM(input_size=input_size[3], hidden_size=hidden_size[3],
                                        num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size[3], out_size)

    def forward(self, input):
        # input(batch_size,seq_len,input_size)
        #self.hidden = self.initHidden(input.size(0))
        #out, _ = self.lstm(input, self.hidden)

        out_lstm_line0, _ = self.lstm_line0(input)

        input1 = out_lstm_line0.reshape(input.size(0), -1, self.input_size[1])
        out_lstm_line1, _ = self.lstm_line1(input1)

        input2 = out_lstm_line1.reshape(input.size(0), -1, self.input_size[2])
        out_lstm_line2, _ = self.lstm_line2(input2)

        input3 = out_lstm_line2.reshape(input.size(0), -1, self.input_size[3])
        out_lstm_line3, _ = self.lstm_line3(input3)

        output = self.linear(out_lstm_line3[:, -1, :].squeeze())

        #out, _ = self.lstm(input)
        #temp = out[:, -1, :].squeeze()
        #output = self.linear(temp)
        #print(temp.size(), output.size())
        return output