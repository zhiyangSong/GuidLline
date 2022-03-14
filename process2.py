import matplotlib.pyplot as plt
import numpy as np

def plotMap(dataDir, segmentNum, tra_length):
    tra = np.loadtxt("./data/bag_2/tra.csv", delimiter=",", dtype="double")
    for index in range(segmentNum):
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

        plt.plot(tra[:tra_length, 0], tra[:tra_length, 1], color='g')   # 轨迹

        # plt.plot(xpoint, ypoint, color='r')   # 中心线
        plt.plot(l_b_x, l_b_y, color='b')
        plt.plot(r_b_x, r_b_y, color='b')

    plt.show()


dataDir = "./data/bag_2/"
segmentNum = 3
plotMap(dataDir, segmentNum, 999)