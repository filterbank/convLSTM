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
        self.lstm_line1 = torch.nn.LSTM(input_size=input_size[0], hidden_size=hidden_size[1],
                                        num_layers=num_layers, batch_first=True)
        '''
        self.lstm_line2 = torch.nn.LSTM(input_size=input_size[0], hidden_size=hidden_size[2],
                                        num_layers=num_layers, batch_first=True)
        '''

        self.lstm_horiz0 = torch.nn.LSTM(input_size=input_size[1], hidden_size=hidden_size[0],
                                         num_layers=num_layers, batch_first=True)
        self.lstm_horiz1 = torch.nn.LSTM(input_size=input_size[1], hidden_size=hidden_size[1],
                                         num_layers=num_layers, batch_first=True)
        '''
        self.lstm_horiz2 = torch.nn.LSTM(input_size=input_size[1], hidden_size=hidden_size[2],
                                        num_layers=num_layers, batch_first=True)
        '''


        '''
        self.linear_horiz = nn.Linear(hidden_size, out_size)
        self.linear_line = nn.Linear(hidden_size, out_size)
        '''

        self.linear = nn.Linear(2*hidden_size[1], out_size)

    def forward(self, input):
        # input(batch_size,seq_len,input_size)
        #self.hidden = self.initHidden(input.size(0))
        #out, _ = self.lstm(input, self.hidden)
        '''
        out_lstm_horiz, _ = self.lstm_horiz(input)
        out_linear_horiz = self.linear_horiz(out_lstm_horiz[:, -1, :].squeeze())

        out_lstm_verti, _ = self.lstm_verti(input.reshape(input.size(0),-1,self.input_size[1]))
        out_linear_verti = self.linear_verti(out_lstm_verti[:, -1, :].squeeze())

        out_lstm_line, _ = self.lstm_line(input.reshape(input.size(0),-1,self.input_size[2]))
        out_linear_line = self.linear_line (out_lstm_line[:, -1, :].squeeze())

        output = self.linear(torch.cat((out_linear_horiz, out_linear_verti, out_linear_line),-1))

        '''
        out_lstm_line0, _ = self.lstm_line0(input)
        out_lstm_line1, _ = self.lstm_line1(out_lstm_line0[:, -1, :].reshape(input.size(0), -1, self.input_size[0]))

        #out_lstm_line2, _ = self.lstm_line2(out_lstm_line1[:, -1, :].reshape(input.size(0), -1, self.input_size[0]))


        out_lstm_horiz0, _ = self.lstm_horiz0(input.reshape(input.size(0), -1, self.input_size[1]))
        out_lstm_horiz1, _ = self.lstm_horiz1(out_lstm_horiz0[:, -1, :].reshape(input.size(0), -1, self.input_size[1]))
        #out_lstm_horiz2, _ = self.lstm_horiz2(out_lstm_horiz1[:, -1, :].reshape(input.size(0), -1, self.input_size[1]))


        output = self.linear(torch.cat((out_lstm_line1[:, -1, :].squeeze(dim=-2), out_lstm_horiz1[:, -1, :].squeeze(dim=-2)), -1))

        #out, _ = self.lstm(input)
        #temp = out[:, -1, :].squeeze()
        #output = self.linear(temp)
        #print(temp.size(), output.size())
        return output.squeeze()

'''
class PredictorLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(PredictorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_line0 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size[0],
                                        num_layers=num_layers, batch_first=True)

        self.lstm_line1 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size[1],
                                       num_layers=num_layers, batch_first=True)

        self.lstm_line2 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size[2],
                                       num_layers=num_layers, batch_first=True)

        self.lstm_line3 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size[3],
                                        num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size[3], out_size)

    def forward(self, input):
        # input(batch_size,seq_len,input_size)
        #self.hidden = self.initHidden(input.size(0))
        #out, _ = self.lstm(input, self.hidden)

        out_lstm_line0, _ = self.lstm_line0(input)

        input1 = out_lstm_line0[:, -1, :].reshape(input.size(0),-1,self.input_size)
        out_lstm_line1, _ = self.lstm_line1(input1)

        input2 = out_lstm_line1[:, -1, :].reshape(input.size(0), -1, self.input_size)
        out_lstm_line2, _ = self.lstm_line2(input2)

        input3 = out_lstm_line2[:, -1, :].reshape(input.size(0), -1, self.input_size)
        out_lstm_line3, _ = self.lstm_line3(input3)

        output = self.linear(out_lstm_line3[:, -1, :].squeeze())

        #out, _ = self.lstm(input)
        #temp = out[:, -1, :].squeeze()
        #output = self.linear(temp)
        #print(temp.size(), output.size())
        return output
'''