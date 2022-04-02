
import math
import numpy as np
import matplotlib.pyplot as plt

def uniformization(tra, len, show=False):
    """
    把密度不均匀的轨迹点均匀化（两点之间距离相近）
    tra: 轨迹点. shape: (N, 2|4)
    len: 相两点之间的距离
    """
    # tra = np.loadtxt("{}tra.csv".format(traDir), delimiter=",", dtype="double")
    n = tra.shape[0]
    i = 0
    saveList = []
    saveList.append(tra[i, :])  # 添加第一个点
    for j in range(1, n):
        # (x1-x2)**2 + (y1-y2)**2
        dis = math.sqrt((tra[i, 0]-tra[j, 0])**2 + (tra[i, 1]-tra[j, 1])**2)
    
        if dis > len:
            saveList.append(tra[j, :])
            # print("dis = {}".format(dis))
            i = j
    saveList = np.array(saveList)
    # print("saveList len: ", saveList.shape)
    if show:
        plt.scatter(saveList[:, 0], saveList[:, 1])
        plt.show()
    return saveList




# 轨迹点抽稀的方法，弃用
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