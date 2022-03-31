import os 
from process import  plotMap,  getTrainData ,getTrainData2 ,getTrainData3
import numpy as np
import glob

# dataDir = './data3/junction/'
# dataDir = "./data/bag_20220211_2/"
# traDir = "./data3/bag_20220211_2/"
# path_file_number=glob.glob(pathname='{}segment*.csv'.format(dataDir))
# # print(path_file_number)
# plotMap(dataDir, segBegin=0, segEnd=0)

# # plotLane(dataDir)

# fectures, labels = getTrainData(traDir , dataDir, limit_1=-850, limit_2=-700)
# # fectures, labels = getTrainData(dataDir, limit_1=-200, limit_2=-100)
# print(fectures.shape)
# print(labels.shape)


'''
批量处理数据'''
if not os.path.exists("./data_input"):
    os.makedirs("./data_input")
fec = []
lab = []
# 第一个路口数据处理
traDir  = "./data"
juncDir = './data/junction'
path_data_number = glob.glob(pathname = '{}/bag_2022*'.format(traDir))
for i in path_data_number:
    print(i)
    fectures, labels = getTrainData2(juncDir = juncDir,traDir = i+'/',limit_1=-200, limit_2=-100)
    fec.append(fectures)
    lab.append(labels)

# 路口3数据处理
traDir  = "./data3"
juncDir = './data3/junction'
path_data_number3 = glob.glob(pathname = '{}/bag_2022*'.format(traDir))
for i in path_data_number3:
    print(i)
    fectures, labels = getTrainData2(juncDir = juncDir,traDir =i+"/",  limit_1=-850, limit_2=-700)
    fec.append(fectures)
    lab.append(labels)

# 处理完的模型输入tensor ,按照格式保存到data_input中
fec = np.array(fec).flatten().reshape(len(path_data_number)+len(path_data_number3) , -1)
print(fec.shape)
lab = np.array(lab).flatten().reshape(len(path_data_number)+len(path_data_number3), -1)
np.save("{}features".format("./data_input/"), fec)
np.save("{}labels".format("./data_input/"), lab)



# 看一下模型的输入数据
dataDir = './data_input'
fec = np.load("{}/features.npy".format(dataDir))
lab = np.load("{}/labels.npy".format(dataDir))
print("feature : {}".format( fec))
print(fec.shape)
print(lab.shape)