import os
from matplotlib import pyplot as plt 
from process import *
import numpy as np
import glob
from process_data.B_Spline_Approximation import BS_curve
from process_data.uniformization import uniformization

def fun():
    """
    把各个路段数据整合
    """

    fea_1 = np.load('./data_input/features_1.npy')
    # fea_2 = np.load('./data_input/features_2.npy')
    fea_3 = np.load('./data_input/features_3.npy')
    features = np.vstack([fea_1, fea_3])
    print(features.shape)
    np.save('./data_input/features', features)

    lab_1 = np.load('./data_input/labels_1.npy')
    # lab_2 = np.load('./data_input/labels_2.npy')
    lab_3 = np.load('./data_input/labels_3.npy')
    labels = np.vstack([lab_1 ,lab_3])
    print(labels.shape)
    np.save('./data_input/labels', labels)
    print("fea shape: ", features.shape, " lab shape: ", labels.shape)




limitConfig = {
    "data_0": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1] ,    # y 轴坐标
    "data_6": [-826, -726, 0]     # x 轴坐标
}



if __name__ == '__main__':
  
    dataDir = "./data0"
    limit = limitConfig['data_0']
    LCDirec = 'left'        # 左边换道
    juncDir = "./data0/junction"
    traDir = "./data0/bag_20220110_2"
    index = 0

 
     # 数据预处理
    preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)
    print("预处理完成，道路边界信息存在junction的boundary.npy,轨迹信息存在每个包的tra.npy,中心线信息存在centerLane.npy")

    # 打印轨迹，相对坐标，即简单的将轨迹和道路边界线规整到以轨迹起点为（0，0）
    # pltTra(dataDir=dataDir, juncDir=juncDir, traDir=traDir)  
     # 变换坐标使得车辆轨迹的第一个点的朝 x 轴正向
    # newTra, newBound = transfor(juncDir=juncDir, traDir=traDir, show=True)
    # 数据处理，生成fea ,lab 的npy文件
    # 单条数据处理
    # fea, lab = getTrainData(tra=newTra, boundary=newBound)
    # print(fea)
    # print(lab)
    #对所有数据处理
    fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index)
    print("fea shape: ", fea.shape, " lab shape: ", lab.shape)

   
    # dataDir = "./data2"
    # limit = limitConfig['data_2']
    # LCDirec = 'left'        # 左边换道
    # juncDir = "./data/junction"
    # traDir = "./data/bag_20220110_2"
    # index = 2

    # preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)
    # print("预处理完成，道路边界信息存在junction的boundary.npy,轨迹信息存在每个包的tra.npy")
    # fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index)
    # print("fea shape: ", fea.shape, " lab shape: ", lab.shape)

    
  

    dataDir = "./data6"
    limit = limitConfig['data_6']
    LCDirec = 'left'        # 左边换道
    juncDir = "./data6/junction"
    traDir = "./data6/bag_20220121_1"
    index = 6


    # # 打印轨迹，相对坐标，即简单的将轨迹和道路边界线规整到以轨迹起点为（0，0）
    # pltTra(dataDir=dataDir, juncDir=juncDir, traDir=traDir)  


    # 数据预处理
    preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)
    print("预处理完成，道路边界信息存在junction的boundary.npy,轨迹信息存在每个包的tra.npy")

    # # # 变换坐标使得车辆轨迹的第一个点的朝 x 轴正向
    # # newTra, newBound = transfor(juncDir=juncDir, traDir=traDir, show=True)
    # # # 数据处理，生成fea ,lab 的npy文件
    # # # 单条数据处理
    # # fea, lab = getTrainData(tra=newTra, boundary=newBound)
    # # print(fea)
    # # print(lab)
    # #对所有数据处理
    fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index)
    print("fea shape: ", fea.shape, " lab shape: ", lab.shape)

    # # 将_input中多个路段数据合并
    fun()

  









