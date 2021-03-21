import argparse
import os
import torch
import csv
import sys
import copy
from util import *
from torch import nn, optim
from dataset import KpApF107Dataset
from model import PredictorLSTM
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()  # copy from train.py
parser.add_argument('--dataset_path', type=str, default='../data/f10.7/test2015_2018.csv', help='path of the dataset')

#****************************************#
#copy from train.py
parser.add_argument('--data_type', type=str, default='F10.7', help='KP|AP|F10.7')
parser.add_argument('--time_step', type=int, default= 27*2, help=' it indicates day')
parser.add_argument('--pred_len', type=int, default= 3, help=' it indicates predicted day')

parser.add_argument('--input_size', type=int, default= (1,27), help='the length of every timestep')
parser.add_argument('--hidden_size', type=int, default=(27*4, 27*2), help='the output length of every timestep ')
parser.add_argument('--num_layers', type=int, default=1, help='the number of lstm layer(vertical)')
parser.add_argument('--out_size', type=int, default=3, help='the length of predicted sequence')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='dir of save file and model')

#****************************************#
parser.add_argument('--test_type', type=str, default= 'pred_len', help='predlen| pred_total_len')
parser.add_argument('--pred_total_len', type=int, default= 6, help=' it indicates predicted day and must be 27')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches and must be 1')
parser.add_argument('--load_epoch', type=str, default='latest', help='load model ')
opt = parser.parse_args()
print(opt)

load_model_dir = os.path.join(opt.checkpoint_dir, 'model') #copy from train.py
save_file_dir = os.path.join(opt.checkpoint_dir, 'ckpt') #copy from train.py
test_results_path = os.path.join(save_file_dir, '%s_test_results_%s.csv' % (opt.load_epoch, opt.data_type))


model = PredictorLSTM(opt.input_size, opt.hidden_size, opt.num_layers, opt.out_size)  # 27 *8
criterion = nn.MSELoss()

use_gpu = True if torch.cuda.is_available() else False
if use_gpu:
    model = model.cuda()

model = load_model(model, epoch=opt.load_epoch, data_type=opt.data_type, load_model_dir=load_model_dir)

if opt.test_type == 'pred_total_len':
    test_dataset = KpApF107Dataset(opt.dataset_path, opt.time_step, opt.pred_total_len, opt.data_type)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    testmes = test_evaluate_model(model, test_loader, criterion, pred_len=opt.pred_len, test_results_path=test_results_path)

if opt.test_type == 'pred_len':
    test_dataset = KpApF107Dataset(opt.dataset_path, opt.time_step, opt.pred_len, opt.data_type)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    testmes = evaluate_model(model, test_loader, criterion, test_results_path=test_results_path)

plot_curve(test_results_path, opt.data_type, save_file_dir, load_epoch=opt.load_epoch)
#### write testmes


