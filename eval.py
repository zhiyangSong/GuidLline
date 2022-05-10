import numpy as np

from BCModel.arguments import get_common_args
from BCModel.net import BCNet

import torch

from matplotlib import pyplot as plt 

from process_data.B_Spline_Approximation import  BS_curve
from process import *
from test import config


args = get_common_args()


def eval(feature, label, juncDir, traDir, modelPath, cpNum, degree):
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

    # 将label  和pred  都转为numpy
    label = label.reshape(-1, 2)
    pred = pred.view(-1, 2).detach().numpy()
    print("pred :{}".format(pred))
    print("label : {}".format(label))
    loss = np.sum(pred-label)**2
    print("loss is: ", loss)

    centerLane = np.load("{}/centerLane.npy".format(juncDir))
    cos = centerLane[0, 2]
    sin = centerLane[0, 3]
    
    bs = BS_curve(n=cpNum, p=degree)        # 初始化B样条
    bs.get_knots()                          # 计算b样条节点并设置

    # firstCP = np.array(label[0, :])
    # pred = np.vstack([firstCP, pred])
    pred[0, :] = label[0, :]

    x_asis = np.linspace(0, 1, 101)
    #设置控制点
    bs.cp = label       # 标签(控制点)
    curves_label = bs.bs(x_asis)
    bs.cp = pred        # 网络输出
    curves_pred = bs.bs(x_asis)

    rot_label = rot(tra=curves_label, point=[0, 0], sin=sin, cos=cos, rotDirec=0)
    rot_pred = rot(tra=curves_pred, point=[0, 0], sin=sin, cos=cos, rotDirec=0)
    rot_label_cp = rot(tra=label, point=[0, 0], sin=sin, cos=cos, rotDirec=0)
    rot_pred_cp = rot(tra=pred, point=[0, 0], sin=sin, cos=cos, rotDirec=0)
    rot_label[:, 0] *= 10
    rot_pred[:, 0] *= 10
    rot_label_cp[:, 0] *= 10
    rot_pred_cp[:, 0] *= 10
    curves_label = rot(tra=rot_label, point=[0, 0], sin=sin, cos=cos, rotDirec=1)
    curves_pred = rot(tra=rot_pred, point=[0, 0], sin=sin, cos=cos, rotDirec=1)
    label_cp = rot(tra=rot_label_cp, point=[0, 0], sin=sin, cos=cos, rotDirec=1)
    pred_cp = rot(tra=rot_pred_cp, point=[0, 0], sin=sin, cos=cos, rotDirec=1)
    
    curves_label[:, 0] += centerLane[0, 0]
    curves_label[:, 1] += centerLane[0, 1]
    curves_pred[:, 0] += centerLane[0, 0]
    curves_pred[:, 1] += centerLane[0, 1]

    label_cp[:, 0] += centerLane[0, 0]
    label_cp[:, 1] += centerLane[0, 1]
    pred_cp[:, 0] += centerLane[0, 0]
    pred_cp[:, 1] += centerLane[0, 1]

    plt.plot(curves_label[:, 0], curves_label[:, 1], color='b')
    plt.plot(curves_pred[:, 0], curves_pred[:, 1], color='r')
    plt.scatter(label_cp[:, 0], label_cp[:, 1], color='b')
    plt.scatter(pred_cp[:, 0], pred_cp[:, 1], color='r')
    
    plotMap(juncDir=juncDir)    # 打印路段信息
    plt.show()


def evalModel(modelPath):
    data_dirs=glob.glob(pathname='./data/*data*')
    print(data_dirs)
    for dir in [ './data/data_2_test' ]:
    # data_dirs=glob.glob(pathname='./data/*data*')
    # for dir in data_dirs:
        sub_data = dir.split('/')[2]
        bagName = config[sub_data]['testBag']
        juncDir = '{}/junction'.format(dir)
        traDir = '{}/{}'.format(dir, bagName)
        print(traDir)

        centerLane = np.load("{}/centerLane.npy".format(juncDir))
        point = [centerLane[0, 0], centerLane[0, 1]]
        cos = centerLane[0, 2]
        sin = centerLane[0, 3]



        preProcess(dataDir=dir , limit = config[sub_data]['limit'],
                    LCDirec=config[sub_data]['LCDirec'])

        tra = np.load("{}/tra.npy".format(traDir))
        start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
        newTra = rot(tra[:, :2], point=point, sin=sin, cos=cos)
        boundary = np.load("{}/boundary.npy".format(juncDir))
        newBoundary = rot(boundary, point=point, sin=sin, cos=cos)
        # TODO 数据抽稀
        re = Reduce(pointNum=20)
        reduce_tra = re.getReducePoint(newTra)
        reduce_bound = re.getReducePoint(newBoundary)
        assert reduce_tra.shape[0] == 20, \
        "抽稀后的数据点个数要等于 pointNum"
        assert reduce_bound.shape[0] == 20, \
            "抽稀后的数据点个数要等于 pointNum"
        # TODO 把 x 轴缩放
        reduce_tra[:, 0] /= 10.
        reduce_bound[:, 0] /= 10.

        # 计算源数据的 fea 和 lab并保存，用作效果评估
        fea, lab = buildTrainData(
                reduce_tra=reduce_tra, reduce_bound=reduce_bound, cos=cos,
                sin=sin, start_speed=start_speed, rotDirec=1 , LCDirec =config[sub_data]['LCDirec'])
        np.save("{}/feature.npy".format(traDir), fea)
        np.save("{}/label.npy".format(traDir), lab)


        











        fea = np.load("{}/feature.npy".format(traDir))
        lab = np.load("{}/label.npy".format(traDir))
        eval(
            feature=fea,
            label=lab,
            modelPath=modelPath,
            juncDir=juncDir, 
            traDir=traDir,
            cpNum=8, degree=3
        )







if __name__ == '__main__':
    # 2204_091800 --> 缩放版本
    modelPath = './model/2204_291736/episodes_499.pth'
    evalModel(modelPath=modelPath)