import os
import copy
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class FlareDataset (Dataset):
    def __init__(self, root, timestep, num_channels = 1, height = 256, width = 256, datatype='flare', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.numfiles = []
        self.files    = []
        self.labels   = []
        subdirs = []
        # read all dictionaries
        for label in [1,0]:
            if label == 1:
                path_name = 'POS'
            else:
                path_name = 'NEG'
            root_sub = os.path.join(root, path_name)
            list = os.listdir(root_sub)

            for i in range(0, len(list)):
                subdir = os.path.join(root_sub, list[i])
                subdirs.append(subdir)
        # rearrange all dictionary
        for i in range(len(subdirs)):
            j = int(random.random() * (i + 1))
            if j <= len(subdirs) - 1:  # 交换
                subdirs[i], subdirs[j] = subdirs[j], subdirs[i]
                #y[i], y[j] = y[j], y[i]
        '''
        for label in [1,0]:
            if label == 1:
                path_name = 'POS'
            else:
                path_name = 'NEG'
            #root_sub = os.path.join(root, path_name)
        '''
        #list = subdirs
        num = 0
        self.numfiles.append(num)
        for subdir in subdirs:
            #subdir = list[i] #subdir = os.path.join(root_sub, list[i])
            if 'POS' in subdir:
                label = 1
            elif 'NEG' in subdir:
                label = 0
            else:
                print('error')
            if os.path.isdir(subdir):
                files = os.listdir(subdir)
                assert len(files) > 0
                for f in files:
                    file = os.path.join(subdir,f)
                    if os.path.splitext(file)[-1] == '.jpg' or os.path.splitext(f)[1] == '.png':
                        num = num + 1
                        #self.files.append(oneday)
                        print(file)
                        self.files.append(file)
                        self.labels.append(label)
                self.numfiles.append(num)

        #self.files = sorted(glob.glob(root + '/*.*'))
        #self.list = list
        self.timestep = timestep
        self.num_chl  = num_channels
        self.img_size = (height, width)
        #self.groupdaynumb = timestep + predlen
        #self.predlen = predlen
        self.datatype = datatype # KP, AP, F10.7

    def __len__(self):
        return len(self.numfiles) - 1


    def __getitem__(self, index):
        groupfilenum = self.timestep #self.numfiles[index]
        groupData = []
        labelData = []
        for i in range(self.numfiles[index], self.numfiles[index + 1]): #i in range(0, groupfilenum):
            file = self.files[i % len(self.files)]
            label= self.labels[i % len(self.files)]
            img = Image.open(file)
            img = img.resize(self.img_size)
            (w, h) = self.img_size
            if self.num_chl == 1:
                img = img.convert('L')

            oneData = np.array(img).reshape(-1, h, w)
            if self.datatype == 'flare':
                groupData.append(oneData)
                labelData.append(label)
            else:
                print('The DataType is wrong !')
        if len(groupData) < groupfilenum * 1:
            len_data = len(groupData)
            oneData = groupData[-1:][0]
            label   = labelData[-1:][0]
            for i in range(0,groupfilenum-len_data):
                groupData.append(oneData)
                labelData.append(label)

        for i in range(0, len(labelData)):
            try:
                assert labelData[i] == labelData[-1]
            except AssertionError:
                print('Assertion Error')

        if  self.datatype == 'flare' and len(groupData) >= groupfilenum * 1:
            labeldata = np.array(labelData[-1:]).astype('float32') #np.array(groupData).astype('float32')
            inputdata = np.array(groupData[:self.timestep]).astype('float32') #must the same for seq_len
        else:
            print(index)
            print('The Group Data is wrong !')
        #print(inputdata.shape, labeldata.shape)
        #return (inputdata-minvalue)/(maxvalue-minvalue), (labeldata - minvalue)/(maxvalue-minvalue), minvalue, maxvalue
        #conv = conv2d(groupfilenum,self.timestep,5)
        #inputdata = conv(inputdata)
        return inputdata, labeldata