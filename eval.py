import imp
import numpy as np

from BCModel.arguments import get_common_args
from BCModel.net import BCNet

import torch
import torch.nn as nn

from matplotlib import pyplot as plt 

from process_data.B_Spline_Approximation import  BS_curve
from process_data.uniformization import uniformization
from process import plotMap
from process import getTrainData


args = get_common_args()



def eval(dataDir, limit_1, limit_2, cpNum, degree, distance):
    """
    查看模型预测效果
    dataDir: 原始数据路径
    limit_1, limit_2: 路段范围
    cpNum, degree: B样条控制点与阶数
    distance: 抽稀距离
    """

    # 加载模型
    path="./model/2203_231423/episodes_1499.pth"
    model = BCNet(args.input_size, args.output_size, args)
    model.load_state_dict(torch.load(path))
    print('load network successed')
    model.eval()

    loss_function = nn.MSELoss()

    # 加载模型的输入和标签
    feacture = np.load("{}features.npy".format(dataDir))
    label = np.load("{}labels.npy".format(dataDir))
    feacture = torch.FloatTensor(feacture).view(1, -1)
    # 预测出的控制点
    pred = model(feacture)

    # 将label  和pred  都转为numpy
    label = label.reshape(-1, 2)
    pred = pred.view(-1, 2).detach().numpy()
    print("pred :{}".format(pred))
    print("label : {}".format(label))

   

    # 初始化B样条
    bs = BS_curve(n=cpNum, p=degree)
    # tra = np.load("{}tra.npy".format(dataDir))[:, :2]
   
    # 拿到轨迹的开始位置
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    tra = tra[(limit_1 < tra[:, 0]) & (tra[:, 0] < limit_2) , 0:2]
    start_x, start_y = tra[0, 0], tra[0, 1] # 开始位置
    
    tra[:, 0] -= tra[0, 0]
    tra[:, 1] -= tra[0, 1]
    tra = uniformization(tra, distance)      # 抽稀
    
    # 计算b样条节点并设置
    bs.get_knots()
   
    x_asis = np.linspace(0, 1, 101)
    #设置控制点
    bs.cp = label       # 标签(控制点)
    curves_label = bs.bs(x_asis)
    curves_label[:, 0] += start_x   # 把数据恢复为地图位置
    curves_label[:, 1] += start_y

    bs.cp = pred        # 网络输出
    curves_pred = bs.bs(x_asis)
    curves_pred[:, 0] += start_x
    curves_pred[:, 1] += start_y

    tra[:, 0] += start_x        # 把轨迹恢复为地图位置
    tra[:, 1] += start_y
    plt.scatter(tra[:, 0], tra[:, 1])
    plt.plot(curves_pred[:, 0], curves_pred[:, 1], color='r')
    plt.plot(curves_label[:, 0], curves_label[:, 1], color='b')


    plotMap(dataDir=dataDir, showTra=False)    # 打印路段信息











# 处理指定包内数据,保存成features,labels
dataDir="./data/bag_20220108_2/"
fectures, labels = getTrainData(dataDir, limit_1=-200, limit_2=-100)
np.save("{}features".format(dataDir), fectures)
np.save("{}labels".format(dataDir), labels)
eval(dataDir, limit_1=-200, limit_2=-100, cpNum=8, degree=3, distance=5)
