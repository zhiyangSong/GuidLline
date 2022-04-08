import imp
import numpy as np

from BCModel.arguments import get_common_args
from BCModel.net import BCNet

import torch
import torch.nn as nn

from matplotlib import pyplot as plt 

from process_data.B_Spline_Approximation import  BS_curve
from process_data.uniformization import uniformization
from process import *


args = get_common_args()



def eval( juncDir, traDir, modelPath, cpNum, degree, distance):
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

    

    # 加载模型的输入和标签
    feacture = np.load("{}/features.npy".format(traDir))
    label = np.load("{}/labels.npy".format(traDir))
    feacture = torch.FloatTensor(feacture).view(1, -1)
    # 预测出的控制点
    pred = model(feacture)

    

    loss_function = nn.MSELoss(reduction = 'sum')
    loss = loss_function(pred, torch.FloatTensor(label).view(1, -1))
    print("loss is: ", loss)

    # 将label  和pred  都转为numpy
    label = label.reshape(-1, 2)
    pred = pred.view(-1, 2).detach().numpy()
    print("pred :{}".format(pred))
    print("label : {}".format(label))
    
    # 从b样条控制点还原成曲线
    bs = BS_curve(n=cpNum, p=degree)        # 初始化B样条 
    bs.get_knots()          # 计算b样条节点并设置
    x_asis = np.linspace(0, 1, 101)
    #设置控制点
    bs.cp = label        
    curves_label = bs.bs(x_asis)




     # 找到旋转依据的点和角度
    tra = np.load("{}/tra.npy".format(traDir))
    point =[ tra[0,0] ,tra[0,1]]
    cos  = tra[0,2] / math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    sin = tra[0,3] / math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    # 将曲线旋转回去
    curves_label = rot(curves_label, point=point, sin=sin, cos=cos, rotDirec=1)   # 旋转
    
   
   





    bs.cp = pred        # 网络输出
    curves_pred = bs.bs(x_asis)
    curves_pred = rot(curves_pred, point=point, sin=sin, cos=cos, rotDirec=1)   # 旋转
   


    plt.plot(curves_pred[:, 0], curves_pred[:, 1], color='r')
    plt.plot(curves_label[:, 0], curves_label[:, 1], color='b')
    # 打印抽稀后的轨迹点
    tra = np.load("{}/tra.npy".format(traDir))
    tra = uniformization(tra, distance)     # 抽稀
    plt.scatter(tra[:, 0], tra[:, 1])


    plotMap(juncDir=juncDir)    # 打印路段信息
    plt.show()






limitConfig = {
    "data_1": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1] ,    # y 轴坐标
    "data_3": [-850, -700, 0]     # x 轴坐标
}
limit = limitConfig["data_1"]
traDir="./data/bag_20220111_4"
juncDir = './data/junction'
LCDirec = 'left'


limit = limitConfig["data_3"]
traDir="./data3/bag_20220110_3"
juncDir = './data3/junction'
LCDirec = 'left'




modelPath = './model/2204_021116/episodes_1999.pth'


newTra, newBound = transfor(juncDir=juncDir, traDir=traDir, show=False)
fectures ,labels = getTrainData(tra=newTra, boundary=newBound)
np.save("{}/features".format(traDir), fectures)
np.save("{}/labels".format(traDir), labels)


eval(  modelPath=modelPath,juncDir=juncDir, traDir=traDir,cpNum=8, degree=3, distance=5)


modelPath = './model/2204_081458/episodes_1999.pth'
newTra, newBound = transfor(juncDir=juncDir, traDir=traDir, show=False)
fectures ,labels = getTrainData(tra=newTra, boundary=newBound)
np.save("{}/features".format(traDir), fectures)
np.save("{}/labels".format(traDir), labels)
eval(  modelPath=modelPath,juncDir=juncDir, traDir=traDir,cpNum=8, degree=3, distance=5)



modelPath = './model/2204_081827/episodes_1999.pth'
newTra, newBound = transfor(juncDir=juncDir, traDir=traDir, show=False)
fectures ,labels = getTrainData(tra=newTra, boundary=newBound)
np.save("{}/features".format(traDir), fectures)
np.save("{}/labels".format(traDir), labels)
eval(  modelPath=modelPath,juncDir=juncDir, traDir=traDir,cpNum=8, degree=3, distance=5)



