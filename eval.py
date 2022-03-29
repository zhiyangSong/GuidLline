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



def eval(feature, label, juncDir, traDir, modelPath, cpNum, degree, distance):
    """
    查看模型预测效果
    juncDir: 车道轨迹路径
    traDir: 车辆轨迹路径
    cpNum, degree: B样条控制点与阶数
    distance: 抽稀距离
    """
    # 加载模型
    model = BCNet(args.input_size, args.output_size, args)
    model.load_state_dict(torch.load(modelPath))
    print('load network successed')
    model.eval()

    feature = torch.FloatTensor(feature).view(1, -1)
    pred = model(feature)

    loss_function = nn.MSELoss()
    loss = loss_function(pred, torch.FloatTensor(label).view(1, -1))
    print("loss is: ", loss)

    # 将label  和pred  都转为numpy
    label = label.reshape(-1, 2)
    pred = pred.view(-1, 2).detach().numpy()
    print("pred :{}".format(pred))
    print("label : {}".format(label))
    
    bs = BS_curve(n=cpNum, p=degree)        # 初始化B样条
   
    # 拿到轨迹的开始位置
    tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
    tra = tra[:, 0:2]
    start_x, start_y = tra[0, 0], tra[0, 1] # 开始位置
    tra[:, 0] -= tra[0, 0]                  # 相对坐标
    tra[:, 1] -= tra[0, 1]
    tra = uniformization(tra, distance)     # 抽稀
    bs.get_knots()                          # 计算b样条节点并设置
   
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
    # 保存预测的轨迹数据
    # np.save("{}/tra_pred".format(traDir), curves_pred)

    tra[:, 0] += start_x        # 把轨迹恢复为地图位置
    tra[:, 1] += start_y
    plt.scatter(tra[:, 0], tra[:, 1])
    plt.plot(curves_pred[:, 0], curves_pred[:, 1], color='r')
    plt.plot(curves_label[:, 0], curves_label[:, 1], color='b')

    # label[:, 0] += start_x
    # label[:, 1] += start_y
    # pred[:, 0] += start_x
    # pred[:, 1] += start_y
    # plt.scatter(label[:, 0], label[:, 1])
    # plt.scatter(pred[:, 0], pred[:, 1])
    plotMap(juncDir=juncDir)    # 打印路段信息


limitConfig = {
    "data_1": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1]     # y 轴坐标
}
limit = limitConfig["data_1"]
traDir="./data/bag_20220108_2"
juncDir = './data/junction'

# limit = limitConfig["data_2"]
# traDir="./data2/bag_20220112_1"
# juncDir = './data2/junction'

modelPath = './model/2203_281659/episodes_999.pth'
tra = np.load("{}/tra.npy".format(traDir))
boundary = np.load("{}/boundary.npy".format(juncDir))
feature, label = getTrainData(tra=tra, boundary=boundary)

eval(
    feature=feature,
    label=label,
    modelPath=modelPath,
    juncDir=juncDir, 
    traDir=traDir,
    cpNum=8, degree=3, distance=3
)
