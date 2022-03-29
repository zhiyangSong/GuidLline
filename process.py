import matplotlib.pyplot as plt
import numpy as np
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
    path_file_number=glob.glob(pathname='{}*.csv'.format(juncDir))
    if segEnd == 0:
        segEnd = len(path_file_number)
    for index in range(segBegin, segEnd):
        filename = '{}segment_{}.csv'.format(juncDir, index)
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
            tra = np.loadtxt("{}tra.csv".format(traDir), delimiter=",", dtype="double")
            if tra_length == 0:
                plt.plot(tra[tra_begin:, 0], tra[tra_begin:, 1], color='r')   # 轨迹
            else:
                tra_end = tra_begin + tra_length
                plt.plot(tra[tra_begin:tra_end, 0], tra[tra_begin:tra_end, 1], color='r')

        plt.plot(xpoint, ypoint, color='g', linestyle='--')   # 中心线
        plt.plot(l_b_x, l_b_y, color='y')
        plt.plot(r_b_x, r_b_y, color='y')

    plt.show()
    plt.clf()




def calcuBoundary(laneInfo):
    """
    输入一路段信息，计算边界轨迹。
    返回(中心线、左边界，右边界)数据 shape:(N, 6)
    """
    # laneInfo = np.loadtxt(laneDir, delimiter=",", dtype="double")
    # 计算边界线
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
    laneInfo = np.vstack([xpoint, ypoint, l_b_x, l_b_y, r_b_x, r_b_y]).T
    # np.save("{}laneInfo".format(laneDir), laneInfo)
    # print(laneInfo.shape)
    return laneInfo


def bsplineFitting(laneInfo, cpNum, degree, distance, show=False):
    """
    使用B样条拟合轨迹点
    cpNum: 控制点个数
    degree: 阶数
    distance: 轨迹点抽取距离
    """
    tra = laneInfo
    bs = BS_curve(cpNum, degree)
    # 获取左边界线拟合参数并简化轨迹点
    boundary = uniformization(tra, distance)
    # 打印边界点
    xx = boundary[: , 0]
    yy = boundary[: , 1]
    
    # print(boundary.shape)
    paras = bs.estimate_parameters(boundary)
    knots = bs.get_knots()
    if bs.check():
        cp = bs.approximation(boundary)
    uq = np.linspace(0,1,101)
    y = bs.bs(uq)
    if show:
        plt.scatter(xx ,yy)
        plt.plot(y[:,0],y[:,1],'r')
        plt.plot(cp[:,0],cp[:,1],'y')
        plt.scatter(cp[:,0],cp[:,1],c = 'y')
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



def showTra(dataDir):
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]
  
    point = reducePoint(point, step=50)
    point = point.T
    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])
    plt.plot(point[0,:],point[1, :], color='r')
    plt.show()

# showTra("./data/bag_2/")



def getTrainData(juncDir, traDir, limit_1, limit_2):
    """
    数据处理流程
    traDir: 车辆轨迹路径
    juncDir: 道路节点轨迹
    limit_1: 下界
    limit_2: 上界
    """
    # 获取监督数据（轨迹的B样条控制点）
    tra = np.loadtxt("{}tra.csv".format(traDir), delimiter=",", dtype="double")
    tra = tra[(limit_1 < tra[:, 0]) & (tra[:, 0] < limit_2) , :]
    temp_x = tra[0, 0]     # 记录轨迹起始点坐标(全局坐标)
    temp_y = tra[0, 1]
    tra[:, 0] -= tra[0, 0]
    tra[:, 1] -= tra[0, 1]
    end_x = tra[-1, 0]      # 轨迹结束相对坐标，(以轨迹初始点(0,0)为起始点)
    end_y = tra[-1, 1]
    start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    np.save("{}tra".format(traDir), tra)
    traCP = bsplineFitting(tra[:, 0:2], cpNum=8, degree=3, distance=5, show=False)
    # print("轨迹拟合控制点： ", traCP)

    # 拼接第一段和第三段数据
    seg_1 = np.loadtxt("{}segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    seg_2 = np.loadtxt("{}segment_2.csv".format(juncDir), delimiter=",", dtype="double")
    laneInfo = np.vstack([seg_1, seg_2])
    # 截取了路段信息（-200， -100）
    laneInfo = laneInfo[(limit_1 < laneInfo[:, 0]) & (laneInfo[:, 0] < limit_2) , :]
    laneInfo[:, 0] -= temp_x
    laneInfo[:, 1] -= temp_y
    np.save("{}laneInfo".format(traDir), laneInfo)
    # 根据中心线与左右边界距离计算道路左右边界点
    laneInfo = calcuBoundary(laneInfo)
    # 拟合道路左边界
    boundaryCP = bsplineFitting(laneInfo[:, 2:4], cpNum=8, degree=3, distance=5, show=False)
    boundaryCP = np.array(boundaryCP).reshape(1, -1)

    fectures = np.array([0, 0, start_speed, end_x, end_y]).reshape(1, -1)
    fectures = np.hstack([fectures, boundaryCP])
    labels = np.array(traCP).reshape(1, -1)
    return fectures, labels



def getTrainData2(juncDir, traDir, limit_1, limit_2):
    """
    增加了规整方向
    增加了fectures 中对路口的标识{0,1}

    数据处理流程
    traDir: 车辆轨迹路径
    juncDir: 道路节点轨迹
    limit_1: 下界
    limit_2: 上界
    """

    
    # 获取监督数据（轨迹的B样条控制点）
    tra = np.loadtxt("{}tra.csv".format(traDir), delimiter=",", dtype="double")
    tra = tra[(limit_1 < tra[:, 0]) & (tra[:, 0] < limit_2) , :]
    temp_x = tra[0, 0]     # 记录轨迹起始点坐标(全局坐标)
    temp_y = tra[0, 1]
    tra[:, 0] -= tra[0, 0]
    tra[:, 1] -= tra[0, 1]
    end_x = tra[-1, 0]      # 轨迹结束相对坐标，(以轨迹初始点(0,0)为起始点)
    end_y = tra[-1, 1]

    # 如果是第一个路口， 需要规整方向
    # 将轨迹的方向规整为同一方向
    if(traDir.split("/b")[0]  == "./data") :
        tra[:, 0] =  -tra[:, 0]
        tra[:, 1] = -tra[:, 1]
        

    start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    np.save("{}tra".format(traDir), tra)
    traCP = bsplineFitting(tra[:, 0:2], cpNum=8, degree=3, distance=5, show=False)
    # print("轨迹拟合控制点： ", traCP)





    # 拼接第一段和第三段数据
    seg_1 = np.loadtxt("{}segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    seg_2 = np.loadtxt("{}segment_2.csv".format(juncDir), delimiter=",", dtype="double")
    laneInfo = np.vstack([seg_1, seg_2])
    # 截取了路段信息（-200， -100）
    laneInfo = laneInfo[(limit_1 < laneInfo[:, 0]) & (laneInfo[:, 0] < limit_2) , :]
    laneInfo[:, 0] -= temp_x
    laneInfo[:, 1] -= temp_y

    # 如果是第一个路口， 需要规整方向
    # 将轨迹的方向规整为同一方向
    if(traDir.split("/b")[0]  == "./data") :
        laneInfo[:, 0] =  -laneInfo[:, 0]
        laneInfo[:, 1] = -laneInfo[:, 1]
        laneInfo[:, 2] = -laneInfo[:, 2]
        laneInfo[:, 3] = -laneInfo[:, 3]



    np.save("{}laneInfo".format(traDir), laneInfo)
    # 根据中心线与左右边界距离计算道路左右边界点
    laneInfo = calcuBoundary(laneInfo)
    # 拟合道路左边界
    boundaryCP = bsplineFitting(laneInfo[:, 2:4], cpNum=8, degree=3, distance=5, show=False)
    boundaryCP = np.array(boundaryCP).reshape(1, -1)

    fectures = np.array([0, 0, start_speed, end_x, end_y]).reshape(1, -1)
    fectures = np.hstack([fectures, boundaryCP])
   
    if(traDir.split("/b")[0]  == "./data"):

        fectures = np.hstack([fectures, np.array([0]).reshape(1, 1)])

    if(traDir.split("/b")[0]  == "./data3"):
        fectures = np.hstack([fectures, np.array([1]).reshape(1, 1)])

    print(fectures.shape)
    # if(traDir.split("/b")[0]  == "./data"):
    #     labels = np.array(traCP).reshape(1, -1)
    #     labels = np.hstack([labels, np.array([0]).reshape(1, 1)])


    # if(traDir.split("/b")[0]  == "./data3"):
    #     labels = np.array(traCP).reshape(1, -1)
    #     labels = np.hstack([labels, np.array([1]).reshape(1, 1)])
    labels = np.array(traCP).reshape(1, -1)
    print(labels.shape)
    return fectures, labels



def getTrainData3(juncDir, traDir, limit_1, limit_2):
    """
    数据处理流程
    增加了规整方向
    增加了fectures 中对路口的标识{0,1}
    删除了labels 的第一个控制点（0，0）
    traDir: 车辆轨迹路径
    juncDir: 道路节点轨迹
    limit_1: 下界
    limit_2: 上界
    """

    
    # 获取监督数据（轨迹的B样条控制点）
    tra = np.loadtxt("{}tra.csv".format(traDir), delimiter=",", dtype="double")
    tra = tra[(limit_1 < tra[:, 0]) & (tra[:, 0] < limit_2) , :]
    temp_x = tra[0, 0]     # 记录轨迹起始点坐标(全局坐标)
    temp_y = tra[0, 1]
    tra[:, 0] -= tra[0, 0]
    tra[:, 1] -= tra[0, 1]
    end_x = tra[-1, 0]      # 轨迹结束相对坐标，(以轨迹初始点(0,0)为起始点)
    end_y = tra[-1, 1]

    # 如果是第一个路口， 需要规整方向
    # 将轨迹的方向规整为同一方向
    if(traDir.split("/b")[0]  == "./data") :
        tra[:, 0] =  -tra[:, 0]
        tra[:, 1] = -tra[:, 1]
        

    start_speed = math.sqrt(tra[0, 2]**2 + tra[0, 3]**2)
    np.save("{}tra".format(traDir), tra)
    traCP = bsplineFitting(tra[:, 0:2], cpNum=8, degree=3, distance=5, show=False)
    # print("轨迹拟合控制点： ", traCP)





    # 拼接第一段和第三段数据
    seg_1 = np.loadtxt("{}segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    seg_2 = np.loadtxt("{}segment_2.csv".format(juncDir), delimiter=",", dtype="double")
    laneInfo = np.vstack([seg_1, seg_2])
    # 截取了路段信息（-200， -100）
    laneInfo = laneInfo[(limit_1 < laneInfo[:, 0]) & (laneInfo[:, 0] < limit_2) , :]
    laneInfo[:, 0] -= temp_x
    laneInfo[:, 1] -= temp_y

    # 如果是第一个路口， 需要规整方向
    # 将轨迹的方向规整为同一方向
    if(traDir.split("/b")[0]  == "./data") :
        laneInfo[:, 0] =  -laneInfo[:, 0]
        laneInfo[:, 1] = -laneInfo[:, 1]
        laneInfo[:, 2] = -laneInfo[:, 2]
        laneInfo[:, 3] = -laneInfo[:, 3]



    np.save("{}laneInfo".format(traDir), laneInfo)
    # 根据中心线与左右边界距离计算道路左右边界点
    laneInfo = calcuBoundary(laneInfo)
    # 拟合道路左边界
    boundaryCP = bsplineFitting(laneInfo[:, 2:4], cpNum=8, degree=3, distance=5, show=False)
    boundaryCP = np.array(boundaryCP).reshape(1, -1)

    fectures = np.array([0, 0, start_speed, end_x, end_y]).reshape(1, -1)
    fectures = np.hstack([fectures, boundaryCP])
   
    if(traDir.split("/b")[0]  == "./data"):

        fectures = np.hstack([fectures, np.array([0]).reshape(1, 1)])

    if(traDir.split("/b")[0]  == "./data3"):
        fectures = np.hstack([fectures, np.array([1]).reshape(1, 1)])

    print(fectures.shape)
    # if(traDir.split("/b")[0]  == "./data"):
    #     labels = np.array(traCP).reshape(1, -1)
    #     labels = np.hstack([labels, np.array([0]).reshape(1, 1)])


    # if(traDir.split("/b")[0]  == "./data3"):
    #     labels = np.array(traCP).reshape(1, -1)
    #     labels = np.hstack([labels, np.array([1]).reshape(1, 1)])
    labels = np.array(traCP).reshape(1, -1)
    labels = np.delete(labels, [0,1], axis=1)

    

    print(labels.shape)
    return fectures, labels


def rot(tra, point, sin, cos):
    """ 顺时针旋转 """
    newTra = np.zeros_like(tra)
    x0, y0 = point[0], point[1]
    newTra[:, 0] = (tra[:, 0]-x0)*cos + (tra[:, 1]-y0)*sin
    newTra[:, 1] = (tra[:, 1]-y0)*cos - (tra[:, 0]-x0)*sin
    return newTra


def transfor(juncDir, traDir, show):
    begin_seg = np.loadtxt("{}segment_0.csv".format(juncDir), delimiter=",", dtype="double")
    point = [begin_seg[0, 0], begin_seg[0, 1]]
    cos = begin_seg[0, 2]
    sin = begin_seg[0, 3]

    
    tra = np.load("{}tra.npy".format(traDir))
    newTra = rot(tra, point=point, sin=sin, cos=cos)
  
    if show:
        # 绘制旋转后的路段信息
        fileDirs = glob.glob(pathname = '{}segment*.'.format(juncDir))
        for file in fileDirs:
            lane = np.load(file)
            centerLine = rot(tra=lane[:, :2], point=point, sin=sin, cos=cos)
            leftLine = rot(tra=lane[:, 2:4], point=point, sin=sin, cos=cos)
            rightLine = rot(tra=lane[:, 4:6], point=point, sin=sin, cos=cos)
            newLane = np.hstack([centerLine, leftLine, rightLine])
            if show:
                plt.plot(newLane[:, 0], newLane[:, 1], color='g', linestyle='--')
                plt.plot(newLane[:, 2], newLane[:, 3], color='b')
                plt.plot(newLane[:, 4], newLane[:, 5], color='b')

        plt.plot(newTra[:, 0], newTra[:, 1], color='r')         # 新的轨迹
      
        # pltTra(juncDir=juncDir, traDir=traDir)                  # 原有的路段信息
        plt.show()

def pltTra(juncDir, dataDir=None, traDir=None):
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
        plt.plot(tra[:, 0], tra[:, 1], color='r')
    else:       # 打印 dataDir 下所有轨迹
        fileDirs = glob.glob(pathname = '{}/bag_2022*_*'.format(dataDir))
        for file in fileDirs:
            tra = np.loadtxt("{}/tra.csv".format(file), delimiter=",", dtype="double")
            plt.plot(tra[:, 0], tra[:, 1], color='r')
    
    fileDirs = glob.glob(pathname = '{}/segment*.npy'.format(juncDir))
    for file in fileDirs:
        lane = np.load(file)
        if traDir:
            lane[:, [0, 2, 4]] -= start_x
            lane[:, [1, 3, 5]] -= start_y
        plt.plot(lane[:, 0], lane[:, 1], color='g', linestyle='--')
        plt.plot(lane[:, 2], lane[:, 3], color='b')
        plt.plot(lane[:, 4], lane[:, 5], color='b')
    if traDir:      # 绘制边界线
        boundary = np.load("{}/boundary.npy".format(juncDir))
        boundary[:, 0] -= start_x
        boundary[:, 1] -= start_y
        plt.plot(boundary[:, 0], boundary[:, 1], color='r')
    plt.show()






# 路口1数据处理
# tradir  = "./data/bag_20220108_1/"
# laneDir = './data/junction/'

# fectures, labels = getTrainData3(juncDir = laneDir,traDir =tradir,limit_1=-200, limit_2=-100)

# 路口3数据处理
# tradir  = "./datatest/bag_20220118_4/"
# laneDir = './datatest/junction/'
# fectures, labels = getTrainData2(juncDir = laneDir,traDir =tradir,  limit_1=-850, limit_2=-700)

tradir  = "./data/bag_20220108_1/"
laneDir = './data/junction/'
fectures, labels = getTrainData3(juncDir = laneDir,traDir =tradir,limit_1=-200, limit_2=-100)
transfor(juncDir = laneDir  , traDir= tradir , show = True  )




