import argparse
import os
import torch
import csv
import sys
import copy
from util import *
from torch import nn, optim
#from dataset import KpApF107Dataset
from datasetflare import FlareDataset
from LSTMmodel import PredictorLSTM
from convlstm import ConvLSTM
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/Users/xulong/Downloads/ARDS/', help='dir of the dataset')

parser.add_argument('--data_type', type=str, default='flare', help='KP|AP|F10.7|Oil')
parser.add_argument('--time_step', type=int, default= 24, help=' it indicates day')
parser.add_argument('--pred_len', type=int, default= 1, help=' it indicates predicted day')

parser.add_argument('--img_height', type=int, default= 256, help='the height of input image')
parser.add_argument('--img_width',  type=int, default= 256, help='the width  of input image')
parser.add_argument('--input_size', type=int, default= (198,30, 20, 10), help='the length of every timestep')
parser.add_argument('--hidden_size', type=int, default=(30, 20, 10, 10), help='the output length of every timestep ')
parser.add_argument('--in_channels', type=int, default= 1, help='the channels of input image')
parser.add_argument('--kernel_size', type=int, default=(5,5), help='the kernel size')
parser.add_argument('--num_layers', type=int, default=4, help='the number of lstm layer(vertical)')
parser.add_argument('--out_size', type=int, default=1, help='the length of predicted sequence')

parser.add_argument('--batch_size', type=int, default=12, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--milestones', type=list, default=[15,25], help=' setup MultiStepLR ')
#parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_epoch', type=int, default=35, help='number of epochs of training')

#parser.add_argument('--start_test_epoch', type=int, default=50, help='epoch of start testing during training')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='dir of save file and model')
parser.add_argument('--checkpoint_batch', type=int, default=10, help='10 times in a epoch')
#parser.add_argument('--checkpoint_epoch', type=int, default=1, help=' the number of interval epoch (save model and print loss)')

parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)


save_model_dir = os.path.join(opt.checkpoint_dir, 'model')
if os.path.exists(save_model_dir) is False:
    os.makedirs(save_model_dir)

save_file_dir = os.path.join(opt.checkpoint_dir, 'ckpt')
if os.path.exists(save_file_dir) is False:
    os.makedirs(save_file_dir)

test_mse_path = os.path.join(save_file_dir, 'test_mse.csv')
'''
if os.path.exists(test_mse_path) is True:
    os.remove(test_mse_path)
'''
train_path = 'HMI'
test_path  = 'MDI'
train_data_path = os.path.join(opt.dataset_dir, train_path)
test_data_path  = os.path.join(opt.dataset_dir, test_path)

# Configure dataloaders
transforms_ = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
train_dataset = FlareDataset(train_data_path, opt.time_step, opt.in_channels, opt.img_height, opt.img_width, opt.data_type, transforms_=transforms_)
test_dataset  = FlareDataset(test_data_path,  opt.time_step, opt.in_channels, opt.img_height, opt.img_width, opt.data_type, transforms_=transforms_)


#train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,  num_workers=opt.n_cpu)
test_loader  = DataLoader(test_dataset,  batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

model = ConvLSTM(opt.in_channels, 2*opt.in_channels, opt.kernel_size, opt.num_layers,
                 batch_first=False, bias=True, return_all_layers=False)
#model = PredictorLSTM(opt.input_size, opt.hidden_size, opt.num_layers, opt.out_size)  # 27 *8
use_gpu = True if torch.cuda.is_available() else False
if use_gpu:
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr,  betas=(opt.b1, opt.b2))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)

header = ['epoch/total_epoch', 'test_mse']
with open(test_mse_path, 'w') as testcsvmes: # open trainfile
    writertest = csv.writer(testcsvmes)
    writertest.writerow(header)
    # trainning
    for epoch in range(1, opt.n_epoch + 1):
        print('\repoch {}'.format(epoch))
        scheduler.step()
        print('*' * 10)
        model.train()
        for i, (input, label) in enumerate(train_loader):
            if use_gpu:
                input = torch.FloatTensor(input).cuda()
                label = torch.FloatTensor(label).cuda()
            else:
                input = torch.FloatTensor(input)
                label = torch.FloatTensor(label)

            #label = label.squeeze()
            # foreward
            pred = model(input)
            loss = criterion(pred, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i +1) % opt.checkpoint_batch ==0:
                print("[Epoch %d/%d] [Batch %d/%d] [MSEloss: %f] " %
                                     (epoch, opt.n_epoch, i+1, len(train_loader), loss))

            #sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [MSEloss: %f] " %
            #                     (epoch, opt.n_epoch, i+1, len(train_loader), loss/opt.batch_size))
        # save model
        save_model(model, epoch=epoch, data_type=opt.data_type, save_model_dir=save_model_dir)
        save_model(model, epoch='latest', data_type=opt.data_type, save_model_dir=save_model_dir)
        # evaluate model in testdata
        #testmes = evaluate_model(model, test_loader, criterion)
        #print(testmes[0])
        #writertest.writerow([str(epoch)+'/'+str(opt.n_epoch), testmes[0]])

    #plot_curve(test_results_path, datatype = opt.data_type, savedir = test_results_dir)






