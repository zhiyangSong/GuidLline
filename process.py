from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import glob

def plotMap(dataDir, segBegin=0, segEnd=0, tra_begin=0, tra_length=0):
    """
    tra_begin: 需要的打印轨迹的起始点
    tra_length: 需要打印的轨迹长度。0表示到结束
    """
    # 获取路径下文件夹下个数
    path_file_number=glob.glob(pathname='{}*.csv'.format(dataDir))
    if segEnd == 0:
        segEnd = len(path_file_number) - 1
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    for index in range(segBegin, segEnd):
        filename = '{}segment_{}.csv'.format(dataDir, index)
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
        if tra_length == 0:
            plt.plot(tra[tra_begin:, 0], tra[tra_begin:, 1], color='r')   # 轨迹
        else:
            tra_end = tra_begin + tra_length
            plt.plot(tra[tra_begin:tra_end, 0], tra[tra_begin:tra_end, 1], color='r')

        plt.plot(xpoint, ypoint, color='g', linestyle='--')   # 中心线
        plt.plot(l_b_x, l_b_y, color='b')
        plt.plot(r_b_x, r_b_y, color='b')

    plt.show()


# dataDir = "../prediction/bag/"
# dataDir = "./data/bag_2/"
# plotMap(dataDir)

# dataDir = "./data/bag_3/"
# plotMap(dataDir=dataDir, segBegin=7, segEnd=12, tra_begin=2200, tra_length=500)


def reducePoint(tra, step):
    """
    对轨迹点精简处理
    tra: 轨迹点 numpy (Num, 2)
    step: 缩小倍数
    """
    length = tra.shape[0]
    res = []
    for i in range(0, length, step):
        res.append(tra[i, :])
    return np.array(res)


def showTra(dataDir):
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]
  
    point = reducePoint(point, step=50)
    point = point.T
    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])
    plt.plot(point[0,:],point[1, :], color='r')
    plt.show()

showTra("./data/bag_2/")