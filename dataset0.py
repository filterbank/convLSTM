import os
import csv
import xlrd
import copy
import numpy as np
from torch.utils.data import Dataset


class OilDataset (Dataset):

    def __init__(self,filePath, timestep, predlen, datatype='Oil'):

        with xlrd.open_workbook(filePath) as excelfr:
            sheet = excelfr.sheet_by_index(0)
            # print(type(rowall), len(rowall))
            #rowall_list = [row for row in rowall]
            #self.rowall = rowall_list
        self.sheet = sheet
        self.timestep =timestep
        self.groupdaynumb = timestep + predlen
        self.predlen = predlen
        self.datatype = datatype # KP, AP, F10.7

    def __len__(self):
        return self.sheet.nrows - (self.groupdaynumb -1)


    def __getitem__(self, index):

        '''
        len(row)==71
        first is 0
        KP: 12-13, 14-15, 16-17, 18-19, 20-21, 22-23, 24-25, 26-27
        AP: 31-33, 34-36, 37-39, 40-42, 43-45, 46-48, 49-51, 52-54
        F10.7 : 65-69
        '''
        groupData =[]
        groupLabel=[]
        kpstart = 12
        kpstep =2
        kpstepnumb = 8
        apstart = 31
        apstep = 3
        apstepnumb = 8
        f107start = 65
        f107len = 5
        Oilstep = 1
        Oilstepnumb = 198
        Oillen = 198
        Oilstart = 0

        for i in range(index, (index + self.groupdaynumb)):
            #rowday = copy.deepcopy(self.rowall[i])
            rowday = copy.deepcopy(self.sheet.row_values(i,0,Oillen))
            rowone = copy.deepcopy(self.sheet.row_values(i,-1))

            if self.datatype == 'KP':
                for k in range(0, kpstepnumb):
                    groupData.append(rowday[0][kpstart + k * kpstep: kpstart + (k + 1) * kpstep])

            elif self.datatype == 'AP':
                for m in range(0, apstepnumb):
                    groupData.append(rowday[0][apstart + m * apstep: apstart + (m + 1) * apstep])

            elif self.datatype == 'F10.7':
                groupData.append(rowday[0][f107start: f107start + f107len].replace(' ','0'))

            elif self.datatype == 'Oil':
                for m in range(0,Oilstepnumb):
                    groupData.append(rowday[m*Oilstep:(m+1)*Oilstep])
                groupLabel.append(rowone)
            else:
                print('The DataType is wrong !')

        if  self.datatype == 'KP' and len(groupData) == self.groupdaynumb * kpstepnumb:
            labeldata = np.array(groupData[-self.predlen * kpstepnumb:]).astype('float32')
            inputdata = np.array(groupData[:-self.predlen * kpstepnumb]).astype('float32').reshape(-1, kpstepnumb)
            maxvalue= np.array(np.max(inputdata))
            minvalue= np.array(np.min(inputdata))

        elif  self.datatype == 'AP' and len(groupData) == self.groupdaynumb * apstepnumb:
            labeldata = np.array(groupData[-self.predlen * apstepnumb:]).astype('float32')
            inputdata = np.array(groupData[:-self.predlen * apstepnumb]).astype('float32').reshape(-1, apstepnumb)
            maxvalue = np.array(np.max(inputdata))
            minvalue = np.array(np.min(inputdata))

        elif  self.datatype == 'F10.7' and len(groupData) == self.groupdaynumb * 1:
            labeldata = np.array(groupData[-self.predlen * 1:]).astype('float32')
            inputdata = np.array(groupData[:-self.predlen * 1]).astype('float32').reshape(-1, 1)
            maxvalue = np.array(np.max(inputdata))
            minvalue = np.array(np.min(inputdata))
        elif  self.datatype == 'Oil' and len(groupData) == self.groupdaynumb * Oilstepnumb:
            labeldata = np.array(groupLabel[-self.predlen*1:]).astype('float32')
            inputdata = np.array(groupData[:-self.predlen*Oilstepnumb]).astype('float32').reshape(-1, Oilstepnumb)
            maxvalue = np.array(np.max(inputdata))
            minvalue = np.array(np.min(inputdata))
        else:
            print('The Group Data is wrong !')
        #print(inputdata.shape, labeldata.shape)
        #return (inputdata-minvalue)/(maxvalue-minvalue), (labeldata - minvalue)/(maxvalue-minvalue), minvalue, maxvalue
        return (inputdata - minvalue +1) / (maxvalue - minvalue+1), (labeldata - minvalue +1) / (maxvalue - minvalue+1), minvalue-1, maxvalue


if __name__ == '__main__':
    dataset_path = '../data/f10.7/train1968_2014.csv'

    data = KpApF107Dataset(dataset_path, 4, 1, datatype='KP')
    print(data[0])

