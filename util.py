import os
import torch
import csv
import sys
import copy
import math
import numpy as np

import tkinter

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, criterion, test_results_path = './temptest.csv'):
    model.eval()
    eval_loss = 0.
    count_batchsize = 0
    use_gpu = torch.cuda.is_available()  #
    with open(test_results_path, 'w') as testcsvfa:  # open trainfile
        writertest = csv.writer(testcsvfa)
        for  k, (input, label, minvalue, maxvalue) in enumerate(test_loader):
            count_batchsize = count_batchsize + 1

            if use_gpu:
                input = torch.FloatTensor(input).cuda()
                label = torch.FloatTensor(label).cuda()
            else:
                input = torch.FloatTensor(input)
                label = torch.FloatTensor(label)

            label = label.squeeze()

            with torch.no_grad():
                output = model(input)

            loss = criterion(output, label)
            cpu_device = torch.device('cpu')
            if test_results_path !='./temptest.csv': #  write file when csvfpathvalue setup new path (not default path)
                templabel = copy.deepcopy(label).to(cpu_device)
                tempoutput = copy.deepcopy(output).to(cpu_device)
                labellist = templabel.numpy().reshape(-1).tolist()
                outputlist = tempoutput.numpy().reshape(-1).tolist()
                minvalue = minvalue.tolist()
                maxvalue = maxvalue.tolist()
                rowline = labellist + outputlist + minvalue + maxvalue
                '''
                print('\n')
                print(type(label),label.size(), type(output),output.size())
                print(type(labellist),len(labellist),labellist)
                print(type(outputlist),len(outputlist), outputlist)
                print(type(rowline),len(rowline))
                print('\n')
                '''
                writertest.writerow(rowline)
            if test_results_path =='./temptest.csv' and os.path.exists(test_results_path): # delate useless file
                os.remove(test_results_path)
            eval_loss = eval_loss + loss

            if (k+1) % 10 == 0:
                print("[Batch %d/%d] [MSEloss: %f] " % ( k+1, len(test_loader), loss ))

                '''
                sys.stdout.write("\r[Batch %d/%d] [MESloss: %f] " %
                                 ( k+1, len(test_loader), loss / batch_size))
                '''
    print(count_batchsize)
    print('Test Loss: {:.6f}'.format(eval_loss / count_batchsize ))
    mse = (eval_loss / (count_batchsize )).to(cpu_device).numpy().reshape(-1).tolist()
    return mse



def silde_data(slide_input, slide_label, pad_data, slide_flag):
    '''
    :param slide_input: tensor
    :param slide_label: tensor
    :param pad_data: np.numpy
    :param slide_flag: int
    :return: tensor
    '''
    cpu_device = torch.device('cpu')

    slide_len = pad_data.size

    label_tensor_type = slide_label.dtype
    label_device = slide_label.device

    slide_label = slide_label.to(cpu_device).numpy()
    label_shape = slide_label.shape

    temp_label_list = slide_label.reshape(-1).tolist()
    pad_label_list = np.zeros(slide_len).tolist()

    new_label_list = temp_label_list + pad_label_list
    temp_label = np.array(new_label_list[:slide_len]).reshape((label_shape[0], -1))
    label = torch.as_tensor(temp_label, dtype=label_tensor_type, device=label_device)
    temp_slide_label = np.array(new_label_list[slide_len:]).reshape(label_shape)
    new_slide_label = torch.as_tensor(temp_slide_label, dtype=label_tensor_type, device=label_device)


    if slide_flag != 0:
        input_tensor_type = slide_input.dtype
        input_device = slide_input.device

        slide_input = slide_input.to(cpu_device).numpy()
        input_shape = slide_input.shape

        temp_input_list = slide_input.reshape(-1).tolist()
        temp_pad_data = pad_data.reshape(-1).tolist()
        new_input_list = temp_input_list + temp_pad_data
        temp_input = np.array(new_input_list[slide_len:]).reshape(input_shape)
        input = torch.as_tensor(temp_input, dtype=input_tensor_type, device=input_device)
    else:
        input = slide_input

    new_slide_input = input

    return input, label, new_slide_input, new_slide_label


def test_evaluate_model(model, test_loader, criterion, pred_len, test_results_path = './temptest.csv'):
    '''
    this is a function that is only uesd to test, and it expand testing or use of trainned model

    :param model:
    :param test_loader:
    :param pred_len:
    :param criterion:
    :param test_results_path:
    :return:
    '''
    model.eval()
    eval_loss = 0.
    count_batchsize = 0
    use_gpu = torch.cuda.is_available()  #
    with open(test_results_path, 'w') as testcsvfa:  # open trainfile
        writertest = csv.writer(testcsvfa)
        for  k, (slide_input, slide_label, minvalue, maxvalue) in enumerate(test_loader):
            count_batchsize = count_batchsize + 1

            label_list = []
            output_list = []
            group_loss = 0.
            slide_input_data = copy.deepcopy(slide_input)
            slide_label_data = copy.deepcopy(slide_label)

            slide_flag = 0
            pad_data = np.zeros(pred_len)

            print(type(slide_label), type( slide_label.numpy()))
            silde_numb = slide_label.numpy().size // pred_len

            for i in range(silde_numb):

                input, label, slide_input_data, slide_label_data = silde_data(slide_input =slide_input_data,
                                                                              slide_label = slide_label_data,
                                                                              pad_data = pad_data,
                                                                              slide_flag = slide_flag)
                if use_gpu:
                    input = torch.FloatTensor(input).cuda()
                    label = torch.FloatTensor(label).cuda()
                else:
                    input = torch.FloatTensor(input)
                    label = torch.FloatTensor(label)

                label = label.squeeze()

                with torch.no_grad():
                    output = model(input)

                loss = criterion(output, label)

                cpu_device = torch.device('cpu')

                templabel = copy.deepcopy(label).to(cpu_device)
                tempoutput = copy.deepcopy(output).to(cpu_device)
                labellist = templabel.numpy().reshape(-1).tolist()
                outputlist = tempoutput.numpy().reshape(-1).tolist()

                label_list = label_list + labellist
                output_list = output_list + outputlist

                pad_data = tempoutput.numpy().reshape(-1)
                slide_flag = 1

                group_loss = group_loss + loss

                if i == silde_numb-1 and test_results_path !='./temptest.csv' : #  write file when csvfpathvalue setup new path (not default path)

                    minvalue = minvalue.tolist()
                    maxvalue = maxvalue.tolist()
                    rowline = label_list + output_list + minvalue + maxvalue

                    '''
                    print('\n')
                    print(type(label),label.size(), type(output),output.size())
                    print(type(labellist),len(labellist),labellist)
                    print(type(outputlist),len(outputlist), outputlist)
                    print(type(rowline),len(rowline))
                    print('\n')
                    '''
                    writertest.writerow(rowline)

            if test_results_path =='./temptest.csv' and os.path.exists(test_results_path): # delate useless file
                os.remove(test_results_path)

            eval_loss = eval_loss + group_loss

            if (k+1) % 10 == 0:
                print("[Batch %d/%d] [MSEloss: %f] " % ( k+1, len(test_loader), group_loss ))

                '''
                sys.stdout.write("\r[Batch %d/%d] [MESloss: %f] " %
                                 ( k+1, len(test_loader), loss / batch_size))
                '''
    print(count_batchsize)
    print('Test Loss: {:.6f}'.format(eval_loss / count_batchsize ))
    mse = (eval_loss / (count_batchsize )).to(cpu_device).numpy().reshape(-1).tolist()
    return mse


def save_model(model, epoch, data_type = 'F10.7', save_model_dir = './checkpoint/model'):
    """Save all the networks to the disk.

    Parameters:
        epoch (int/lastest) -- current epoch; used in the file name '%s_PredictorLSTM_%s.pth' % (epoch, data_type)
    """
    save_filename = '%s_PredictorLSTM_%s.pth' % (epoch, data_type)
    save_path = os.path.join(save_model_dir, save_filename)
    torch.save(model.state_dict(), save_path)


def load_model(model, epoch = 'latest', data_type = 'F10.7', load_model_dir = './checkpoint/model'):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_PredictorLSTM_%s.pth' % (epoch, data_type)
    """
    load_filename = '%s_PredictorLSTM_%s.pth' % (epoch, data_type)
    load_path = os.path.join(load_model_dir, load_filename)
    model.load_state_dict(torch.load(load_path))
    print('\rThe load the %s model !' %(epoch))
    return model

def plot_curve(csvfpath, datatype, savedir, load_epoch, datanumb = 27*8*2):
    realvalue_list=[]
    predvalue_list=[]
    with open(csvfpath, 'r') as csvfr:
        rowall = csv.reader(csvfr)
        # print(type(rowall), len(rowall))
        # for row in rowall:
        #    print(type(row), len(row), row)
        for row in rowall:
            splitborder = (len(row)-2)//2  # (bachsize =1 , minvalue, maxvalue)
            minvalue = float(row[-2])
            maxvalue = float(row[-1])
            rangevalue = maxvalue - minvalue
            #splitborder =  2
            #print(type(row),len(row), splitborder, row)
            realvaluetemp = row[:splitborder]
            realvalue_list = realvalue_list + [float(x)*rangevalue + minvalue  for x in realvaluetemp]

            predvaluetemp = row[splitborder:-2]
            predvalue_list = predvalue_list + [float(x)*rangevalue + minvalue  for x in predvaluetemp]

            #print(row[:splitborder:])
            #print(realvalue_list)
            #print(row[splitborder:])

    #np.array(list)
    #print(len(realvalue_list))
    for i in range(len(realvalue_list)//datanumb):
        realvalue = np.array(realvalue_list[i*datanumb:(i+1)*datanumb], dtype='float32')
        predvalue = np.array(predvalue_list[i*datanumb:(i+1)*datanumb], dtype='float32')
        print(realvalue.shape, predvalue.shape)
        plot_x = np.arange(i*datanumb, (i+1)*datanumb)
        plt.plot(plot_x, realvalue, 'g', plot_x, predvalue,'r')
        plt.xlabel("Day")
        plt.ylabel(datatype)
        plt.title("%s value Forecast " % (datatype))

        name = '%s_%s_%s_test_results.png' % (i+1, load_epoch, datatype)
        plt.savefig(os.path.join(savedir, name))
        plt.show()
        #plt.ioff()


def plot_curve_dataset(csvfpath, datatype, savedir, daynumb_img = 27*12, imgnumb=10):

    with open(csvfpath, 'r') as csvfr:
        rowall = csv.reader(csvfr)
        # print(type(rowall), len(rowall))
        rowall_list = [row for row in rowall]

    groupData =[]
    kpstart = 12
    kpstep =2
    kpstepnumb = 8
    apstart = 31
    apstep = 3
    apstepnumb = 8
    f107start = 65
    f107len = 5
    f107stepnumb =1

    totaldaynumb = len(rowall_list)
    if imgnumb == 'all' or (totaldaynumb // daynumb_img) <= imgnumb :
        imgnumb = totaldaynumb // daynumb_img

    daynumb = imgnumb * daynumb_img
    for i in range(daynumb):

        rowday = copy.deepcopy(rowall_list[i])

        if datatype == 'KP':
            for k in range(0, kpstepnumb):
                groupData.append(rowday[0][kpstart + k * kpstep: kpstart + (k + 1) * kpstep])

        elif datatype == 'AP':
            for m in range(0, apstepnumb):
                groupData.append(rowday[0][apstart + m * apstep: apstart + (m + 1) * apstep])

        elif datatype == 'F10.7':
            groupData.append(rowday[0][f107start: f107start + f107len].replace(' ','0'))

        else:
            print('The DataType is wrong !')

    if  datatype == 'KP' and len(groupData) == daynumb * kpstepnumb:
        data = np.array(groupData).astype('float32')
        datanumb_img = daynumb_img * kpstepnumb

    elif  datatype == 'AP' and len(groupData) == daynumb * apstepnumb:
        data = np.array(groupData).astype('float32')
        datanumb_img = daynumb_img * apstepnumb

    elif  datatype == 'F10.7' and len(groupData) == daynumb * 1:
        data = np.array(groupData).astype('float32')
        datanumb_img = daynumb_img * f107stepnumb

    else:
        print('The Group Data is wrong !')


    #np.array(list)
    #print(len(realvalue_list))
    for i in range(imgnumb):
        y_value = data[i*datanumb_img:(i+1)*datanumb_img]
        print(y_value.shape)
        plot_x = np.arange(i*datanumb_img, (i+1)*datanumb_img)
        plt.plot(plot_x, y_value, 'g')
        plt.xlabel("Day")
        plt.ylabel(datatype)
        plt.title("%s Real Value " % (datatype))

        name = '%s_%s_real_value.png' % (i+1, datatype)
        #plt.savefig(os.path.join(savedir, name))
        plt.show()

if __name__ == '__main__':
    csvfpath = '../data/f10.7/data1968_2018f107.csv'
    data_type = 'F10.7'
    savedir = './plot_real_value'
    if os.path.exists(savedir) is False:
        os.makedirs(savedir)
    plot_curve_dataset(csvfpath = csvfpath, datatype= data_type, savedir=savedir, daynumb_img=27 * 12, imgnumb='all')

