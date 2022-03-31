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
    "data_1": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1] ,    # y 轴坐标
    "data_3": [-850, -700, 0]     # x 轴坐标
}



if __name__ == '__main__':
  
    dataDir = "./data"
    limit = limitConfig['data_1']
    LCDirec = 'left'        # 左边换道
    juncDir = "./data/junction"
    traDir = "./data/bag_20220110_2"
    index = 1




    # preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)
    # print("预处理完成，道路边界信息存在junction的boundary.npy,轨迹信息存在每个包的tra.npy") 
    # tra, boundary = transfor(juncDir=juncDir, traDir=traDir )
    # fectures ,labels = getTrainData(tra=tra, boundary=boundary)

    # labels = labels.reshape(-1, 2)

    # bs = BS_curve(n=8, p=3)        # 初始化B样条
   

    # # 找到旋转依据的点和角度
    # centerLane = np.load("{}/centerLane.npy".format(juncDir))
    # point = [centerLane[0, 0], centerLane[0, 1]]
    # begin_seg = np.loadtxt("{}/segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    # cos = begin_seg[0, 2]
    # sin = begin_seg[0, 3]
    
    # bs.get_knots()          # 计算b样条节点并设置
    # x_asis = np.linspace(0, 1, 101)
    # #设置控制点
    # bs.cp = labels        
    # curves_label = bs.bs(x_asis)
    # curves_label = rot(curves_label, point=point, sin=sin, cos=cos, rotDirec=1)   # 旋转
    # plt.plot(curves_label[:, 0], curves_label[:, 1], color='b')
    # plotMap(juncDir= juncDir ,traDir = traDir )


 
     # 数据预处理
    preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)
    print("预处理完成，道路边界信息存在junction的boundary.npy,轨迹信息存在每个包的tra.npy")

    # # # 打印轨迹，相对坐标，即简单的将轨迹和道路边界线规整到以轨迹起点为（0，0）
    # # pltTra(dataDir=dataDir, juncDir=juncDir, traDir=traDir)  

    # # 数据处理，生成fea ,lab 的npy文件
    # # 单条数据处理
    # # tra = np.load("{}/tra.npy".format(traDir))
    # # boundary = np.load("{}/boundary.npy".format(juncDir))
    # # fea, lab = getTrainData(tra=tra, boundary=boundary)
    #对所有数据处理
    fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index)
    print("fea shape: ", fea.shape, " lab shape: ", lab.shape)

    # 变换坐标使得车道中心线第一个点的朝 x 轴正向
    # transfor(juncDir=juncDir, traDir=traDir, show=True)


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

    

    dataDir = "./data3"
    limit = limitConfig['data_3']
    LCDirec = 'left'        # 左边换道
    juncDir = "./data3/junction"
    traDir = "./data3/bag_20220121_1"
    index = 3


    # # 打印轨迹，相对坐标，即简单的将轨迹和道路边界线规整到以轨迹起点为（0，0）
    # # pltTra(dataDir=dataDir, juncDir=juncDir, traDir=traDir)  


    # # 数据预处理
    preProcess(dataDir=dataDir, limit=limit, LCDirec=LCDirec)
    print("预处理完成，道路边界信息存在junction的boundary.npy,轨迹信息存在每个包的tra.npy")


    #对所有数据处理
    fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, index=index)
    print(fea)
    print("fea shape: ", fea.shape, " lab shape: ", lab.shape)

    # 将bag_input中多个路段数据合并
    fun()

    # # 变换坐标使得车道中心线第一个点的朝 x 轴正向
    # # transfor(juncDir=juncDir, traDir=traDir, show=True)









