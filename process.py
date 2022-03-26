import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from process_data.uniformization import uniformization, reducePoint
from process_data.B_Spline_Approximation import BS_curve
import math

def plotMap(juncDir, traDir=None, segBegin=0, segEnd=0, tra_begin=0, tra_length=0):
    """
    traDir: 轨迹路径
    juncDir: 道路节点数据路径
    tra_begin: 需要的打印轨迹的起始点
    tra_length: 需要打印的轨迹长度。0表示到结束
    """
    # 获取路径下文件夹下个数
    path_file_number=glob.glob(pathname='{}/*.csv'.format(juncDir))
    if segEnd == 0:
        segEnd = len(path_file_number)
    for index in range(segBegin, segEnd):
        filename = '{}/segment_{}.csv'.format(juncDir, index)
        data = np.loadtxt(filename, delimiter=",", dtype="double")
        xpoint = data[:,0]
        ypoint = data[:,1]
        cos = data[:, 2]
        sin = data[:, 3]
        lLength = data[:, 5]
        rLength = data[:, 7]
        # left boundary
        l_b_x = xpoint - lLength*sin
        l_b_y = ypoint + lLength*cos
        # right boundary
        r_b_x = xpoint + rLength*sin
        r_b_y = ypoint - rLength*cos
        if traDir:      # 如果轨迹路径不为空，则打印轨迹
            tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
            if tra_length == 0:
                plt.plot(tra[tra_begin:, 0], tra[tra_begin:, 1], color='r')   # 轨迹
            else:
                tra_end = tra_begin + tra_length
                plt.plot(tra[tra_begin:tra_end, 0], tra[tra_begin:tra_end, 1], color='r')

        plt.plot(xpoint, ypoint, color='g', linestyle='--')   # 中心线
        plt.plot(l_b_x, l_b_y, color='y')
        plt.plot(r_b_x, r_b_y, color='y')

    plt.show()


def preProcess(dataDir, limit):
    """
    dataDir: 路段数据根目录
    limit: 路段范围 limit[0]: 下界. limit[1]: 上界. limit[2]: 坐标轴
    1: 计算junction路段的边界并保存为 segment_<>.npy 数据
    2: 计算截取后的道路边界信息 -> boundary.npy
    3: 获取dataDir下所有截取范围后的轨迹 tra.npy
    """
    juncDir = "{}/junction".format(dataDir)
    # 1: 计算junction路段的边界并保存为 segment_<>.npy 数据
    fileDirs = glob.glob(pathname='{}/segment*.csv'.format(juncDir))
    for index in range(len(fileDirs)):
        lineDir = "{}/segment_{}.csv".format(juncDir, index)
        segment = np.loadtxt(lineDir, delimiter=",", dtype="double")
        segment = calcuBoundary(segment)
        np.save("{}/segment_{}".format(juncDir, index), segment)

    # 2: 计算截取后的道路边界信息 -> boundary.npy
    seg_1 = np.load("{}/segment_0.npy".format(juncDir))
    seg_2 = np.load("{}/segment_2.npy".format(juncDir))
    boundary = np.vstack([seg_1, seg_2])
    boundary = boundary[(limit[0] < boundary[:, limit[2]]) & (boundary[:, limit[2]] < limit[1]), :]
    if limit[2] == 0:   # 左边界
        np.save("{}/boundary.npy".format(juncDir), boundary[:, 2:4])
    else: np.save("{}/boundary.npy".format(juncDir), boundary[:, 4:6])
    
    # 3: 获取dataDir下所有截取范围后的轨迹 tra.npy
    traFileDirs = glob.glob(pathname='{}/bag_2022*_*'.format(dataDir))
    for traFile in traFileDirs:
        tra = np.loadtxt("{}/tra.csv".format(traFile), delimiter=",", dtype="double")
        tra = tra[(limit[0] < tra[:, limit[2]]) & (tra[:, limit[2]] < limit[1]), :]
        np.save("{}/tra.npy".format(traFile), tra)
        

def pltTra(juncDir, dataDir, traDir=None):
    """
    traDir==None: 打印 dataDir 下所有轨迹
    traDir!=None: 打印一条轨迹（相对坐标）
    """
    if traDir:  # 打印一条轨迹
        tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
        start_x = tra[0, 0]
        start_y = tra[0, 1]
        tra[:, 0] -= start_x
        tra[:, 1] -= start_y
        plt.plot(tra[:, 0], tra[:, 1])
    else:       # 打印 dataDir 下所有轨迹
        fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
        for file in fileDirs:
            tra = np.loadtxt("{}/tra.csv".format(file), delimiter=",", dtype="double")
            plt.plot(tra[:, 0], tra[:, 1])
    
    fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
    for file in fileDirs:
        lane = np.load(file)
        if traDir:
            lane[:, [0, 2, 4]] -= start_x
            lane[:, [1, 3, 5]] -= start_y
        plt.plot(lane[:, 0], lane[:, 1], color='g', linestyle='--')
        plt.plot(lane[:, 2], lane[:, 3], color='b')
        plt.plot(lane[:, 4], lane[:, 5], color='b')
    plt.show()


def calcuBoundary(laneInfo):
    """
    输入一路段信息，计算边界轨迹。
    返回(中心线、左边界，右边界)数据 shape:(N, 6)
    """
    xpoint = laneInfo[:,0]
    ypoint = laneInfo[:,1]
    cos = laneInfo[:, 2]
    sin = laneInfo[:, 3]
    lLength = laneInfo[:, 5]
    rLength = laneInfo[:, 7]
    # left boundary
    l_b_x = xpoint - lLength*sin
    l_b_y = ypoint + lLength*cos
    # right boundary
    r_b_x = xpoint + rLength*sin
    r_b_y = ypoint - rLength*cos
    # laneInfo shape: (dataLength, 6) (中心线、左边界，右边界)
    return np.vstack([xpoint, ypoint, l_b_x, l_b_y, r_b_x, r_b_y]).T


def bsplineFitting(tra, cpNum, degree, distance, show=False):
    """
    使用B样条拟合轨迹点
    cpNum: 控制点个数
    degree: 阶数
    distance: 轨迹点抽取距离
    return: 控制点
    """
    # 获取左边界线拟合参数并简化轨迹点
    traPoint = uniformization(tra, distance)

    bs = BS_curve(cpNum, degree)
    paras = bs.estimate_parameters(traPoint)
    knots = bs.get_knots()
    if bs.check():
        cp = bs.approximation(traPoint)
    x_ticks = np.linspace(0,1,101)
    curves = bs.bs(x_ticks)
    if show:
        plt.scatter(traPoint[:, 0], traPoint[:, 1])
        plt.plot(curves[:, 0], curves[:, 1], color='r')
        plt.plot(cp[:, 0], cp[:, 1], color='y')
        plt.scatter(cp[:, 0], cp[:, 1], color='y')
        plt.show()
    return cp

def polyFitting(laneInfo):
    """
    使用多项式拟合轨迹
    degree: 多项式阶数
    """
    # 获取左边界线拟合参数
    boundary = uniformization(laneInfo[:, 2:4], 5)
    param = np.polyfit(boundary[:, 0], boundary[:, 1], 3)
    plt.scatter(boundary[:, 0], boundary[:, 1])
    x = boundary[:, 0]
    plt.plot(x, param[0]*x**3 + param[1]*x**2 + param[2]*x**1 + param[3], 'k--')
    plt.show()
    return param


def showOneTra(traDir):
    """ 打印一条轨迹 """
    tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
    point = tra[:, :2]
  
    point = uniformization(point, len=5)
    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])
    plt.plot(point[0,:],point[1, :], color='r')
    plt.show()


def getTrainData(traDir, juncDir, limit_1, limit_2, axis=0):
    """
    数据处理流程
    traDir: 车辆轨迹路径
    juncDir: 道路节点轨迹
    limit_1: 下界
    limit_2: 上界
    axis: limit 范围.0: x轴坐标。1: y轴坐标 
    """
    # 获取监督数据（轨迹的B样条控制点）
    tra = np.load("{}/tra.npy".format(traDir))
    temp_x = tra[0, 0]      # 记录轨迹起始点坐标(全局坐标)
    temp_y = tra[0, 1]
    tra[:, 0] -= tra[0, 0]  # 使用相对坐标
    tra[:, 1] -= tra[0, 1]
    end_x = tra[-1, 0]      # 轨迹结束相对坐标，(以轨迹初始点(0,0)为起始点)
    end_y = tra[-1, 1]
    start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    traCP = bsplineFitting(tra=tra[:, 0:2], cpNum=8, degree=3, distance=5, show=False)

    boundary = np.load("{}/boundary.npy".format(juncDir))
    # 拼接第一段和第三段数据
    seg_1 = np.loadtxt("{}/segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    seg_2 = np.loadtxt("{}/segment_2.csv".format(juncDir), delimiter=",", dtype="double")
    laneInfo = np.vstack([seg_1, seg_2])
    laneInfo = laneInfo[(limit_1 < laneInfo[:, axis]) & (laneInfo[:, axis] < limit_2) , :]
    laneInfo[:, 0] -= temp_x
    laneInfo[:, 1] -= temp_y
    # np.save("{}/laneInfo".format(traDir), laneInfo)
    # 根据中心线与左右边界距离计算道路左右边界点
    laneInfo = calcuBoundary(laneInfo)
    # 拟合道路左边界
    boundaryCP = bsplineFitting(laneInfo[:, 2:4], cpNum=8, degree=3, distance=5, show=False)
    boundaryCP = np.array(boundaryCP).reshape(1, -1)

    fectures = np.array([0, 0, start_speed, end_x, end_y]).reshape(1, -1)
    fectures = np.hstack([fectures, boundaryCP])
    labels = np.array(traCP).reshape(1, -1)
    return fectures, labels


def batchProcess(dataDir, juncDir, limit, index):
    '''
    批量处理数据
    limit: 路段范围信息
    index: 路段数据编号
    '''
    if not os.path.exists("./data_input"):
        os.makedirs("./data_input")
    fea = []
    lab = []
    fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
    for file in fileDirs:
        features, labels = getTrainData(
            traDir=file, 
            juncDir=juncDir, 
            limit_1=limit[0], 
            limit_2=limit[1], 
            axis=limit[2]
        )
        fea.append(features)
        lab.append(labels)

    fea = np.array(fea).flatten().reshape(len(fileDirs) , -1)
    lab = np.array(lab).flatten().reshape(len(fileDirs) , -1)
    
    np.save("{}/features_{}".format("./data_input", index), fea)
    np.save("{}/labels_{}".format("./data_input", index), lab)
    return fea, lab


def rotationTra(tra, point, angle):
    """ 输入一条轨迹，返回按 point 逆时针旋转 angle 角度后的轨迹 """
    x0, y0 = point[0], point[1]
    newTra = np.zeros_like(tra)
    newTra[:, 0] = (tra[:, 0]-x0)*np.cos(angle) - (tra[:, 1]-y0)*np.sin(angle)
    newTra[:, 1] = (tra[:, 0]-x0)*np.sin(angle) + (tra[:, 1]-y0)*np.cos(angle)
    return newTra


def augmentData(juncDir, traDir, angle, show=False):
    """
    通过对原始数据根据轨迹起始点为远点旋转不同角度增加数据
    并返回对新创建数据的处理后的numpy格式网络输入
    traDir: 需要增强的数据
    juncDir: 路段信息
    """
    tra = np.loadtxt("{}/tra.csv".format(traDir), delimiter=",", dtype="double")
    newTra = rotationTra(tra[:, :2], point=tra[0, :2], angle=angle)
    print(tra[0, :2])
    fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
    for file in fileDirs:
        lane = np.load(file)
        centerLine = rotationTra(tra=lane[:, :2], point=tra[0, :2], angle=angle)
        leftLine = rotationTra(tra=lane[:, 2:4], point=tra[0, :2], angle=angle)
        rightLine = rotationTra(tra=lane[:, 4:6], point=tra[0, :2], angle=angle)
        newLane = np.hstack([centerLine, leftLine, rightLine])
        if show:
            plt.plot(newLane[:, 0], newLane[:, 1], color='g', linestyle='--')
            plt.plot(newLane[:, 2], newLane[:, 3], color='b')
            plt.plot(newLane[:, 4], newLane[:, 5], color='b')
    if show:
        plt.plot(newTra[:, 0], newTra[:, 1], color='r')
        pltTra(juncDir=juncDir, traDir=traDir)
        plt.show()

