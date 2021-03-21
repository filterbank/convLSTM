import os
import csv
'''
infdir = '/disk96/xxyu/data/kp_ap/kp_ap'
outfdir = '/disk96/xxyu/data/kp_ap/kp_ap'
'''
infdir = './kpapdata'
outfdir = './kpapdata'

#fnames_list = ['train1932_2000.csv', 'test2001_2018.csv']
fnames_list = ['train1968_2014.csv', 'test2015_2018.csv']
trainfpath = os.path.join(outfdir, fnames_list[0])
testfpath = os.path.join(outfdir, fnames_list[1])

if os.path.exists(trainfpath) is True:
    os.remove(trainfpath)
if os.path.exists(testfpath) is True:
    os.remove(testfpath)

list_files = sorted(os.listdir(infdir))
filenumb = len(list_files)       #1932-2018,  1932-2000 trian, 2001-2018 test
#testnumb = 18
testnumb = 1
trainnumb = filenumb - testnumb


with open(trainfpath, 'w') as traincsvfa: # open trainfile
    writertrain = csv.writer(traincsvfa)

    with open(testfpath, 'w') as testcsvfa: # open testfile
        writertest = csv.writer(testcsvfa)

        for i in range(0,filenumb):
            infpath = os.path.join(infdir, list_files[i])
            with open(infpath, 'r') as csvfr:
                rowall = csv.reader(csvfr)
                #print(type(rowall), len(rowall))
                rowall_list = [[row[0][0:71]] for row in rowall]

            print(type(rowall_list), len(rowall_list))
            if i < trainnumb:
                print('train',i , list_files[i])
                for row in rowall_list:
                    writertrain.writerow(row)
            else :
                print('test', i, list_files[i])
                for row in rowall_list:
                    writertest.writerow(row)

print('The generate Dataset program is over !')

