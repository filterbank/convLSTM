import os
import copy
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FlareDataset (Dataset):
    def __init__(self, root, timestep, num_channels = 1, height = 256, width = 256, datatype='flare', transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.numfiles = []
        self.files    = []
        self.labels   = []

        for label in [1,0]:
            if label == 1:
                path_name = 'POS'
            else:
                path_name = 'NEG'
            root_sub = os.path.join(root, path_name)
            list = os.listdir(root_sub)
            num = 0
            self.numfiles.append(num)
            for i in range(0, len(list)):
                subdir = os.path.join(root_sub, list[i])
                if os.path.isdir(subdir):
                    files = os.listdir(subdir)
                    #num = 0
                    for f in files:
                        file = os.path.join(root_sub,subdir,f)
                        if os.path.splitext(file)[1] == '.jpg':
                            num = num + 1
                            #self.files.append(oneday)
                            print(file)
                            self.files.append(file)
                            self.labels.append(label)
                    self.numfiles.append(num)

        #self.files = sorted(glob.glob(root + '/*.*'))
        self.list = list
        self.timestep = timestep
        self.num_chl  = num_channels
        self.img_size = (height, width)
        #self.groupdaynumb = timestep + predlen
        #self.predlen = predlen
        self.datatype = datatype # KP, AP, F10.7

    def __len__(self):
        return len(self.list)


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

        if  self.datatype == 'flare' and len(groupData) > groupfilenum * 1:
            labeldata = np.array(labelData[:self.timestep]).astype('float32') #np.array(groupData).astype('float32')
            inputdata = np.array(groupData[:self.timestep]).astype('float32') #must the same for seq_len
        else:
            print(index)
            print('The Group Data is wrong !')
        #print(inputdata.shape, labeldata.shape)
        #return (inputdata-minvalue)/(maxvalue-minvalue), (labeldata - minvalue)/(maxvalue-minvalue), minvalue, maxvalue
        #conv = conv2d(groupfilenum,self.timestep,5)
        #inputdata = conv(inputdata)
        return inputdata, labeldata