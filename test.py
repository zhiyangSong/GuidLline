import os 
from process import plotMap, plotLane, getTrainData
import numpy as np
import glob

# dataDir = './data/bag_20220211_1/'
# path_file_number=glob.glob(pathname='{}segment*.csv'.format(dataDir))
# print(path_file_number)
# plotMap(dataDir, segBegin=0, segEnd=0)

# plotLane(dataDir)

# getTrainData(dataDir, limit_1=-200, limit_2=-100)


'''
批量处理数据，'''
if not os.path.exists("./data_input"):
    os.makedirs("./data_input")
fec = []
lab = []
datadir  = "./data/"
path_data_number = glob.glob(pathname = '{}bag_2022021*_*'.format(datadir))
for i in path_data_number:
    print(i)
    fectures, labels = getTrainData(i+'/', limit_1=-200, limit_2=-100)
    fec.append(fectures)
    lab.append(labels)

fec = np.array(fec).flatten().reshape(len(path_data_number) , -1)
lab = np.array(lab).flatten().reshape(len(path_data_number) , -1)

np.save("{}features".format("./data_input/"), fec)
np.save("{}labels".format("./data_input/"), lab)





# 看一下模型的输入数据
dataDir = './data_input/'
fec = np.load("{}features.npy".format(dataDir))
print("feature : {}".format( fec))
print(fec.shape)