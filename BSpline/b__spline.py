'''
弃用
'''
import parameter_selection as ps
import numpy as np
import bspline_curve as bc
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from porcess_data.dp import DouglasPeuker


import math

'''
通过给出的一些轨迹点，反求控制点，画出B样条曲线
'''
# 曲线插值法
def curve_inter_figure(D, D_N, k):
    '''
    Input: Data points
    '''
            

    '''
    Step 1. Calculate parameters
    '''
    # p_uniform = ps.uniform_spaced(D_N)
    # print(p_uniform)

    # p_chord_length = ps.chord_length(D_N, D)
    # print(p_chord_length)

    p_centripetal = ps.centripetal(D_N, D)
    # print(p_centripetal)

    '''
    Step 2. Calculate knot vector
    '''
    knot = ps.knot_vector(p_centripetal, k, D_N)
    # print(knot)

    '''
    Step 3. Calculate control points
    '''
    P_inter = bc.curve_interpolation(D, D_N, k, p_centripetal, knot)
    print(P_inter)

    fig = plt.figure()
    for i in range(D_N):
        plt.scatter(D[0][i], D[1][i], color='r')
        plt.scatter(P_inter[0][i], P_inter[1][i], color='b')
       
    for i in range(D_N - 1):
        tmp_x = [P_inter[0][i], P_inter[0][i+1]]
        tmp_y = [P_inter[1][i], P_inter[1][i+1]]
        plt.plot(tmp_x, tmp_y, color='b')

    '''
    Step 4. Calculate the points on the b-spline curve
    '''
    piece_num = 80
    p_piece = np.linspace(0, 1, piece_num)
    P_piece = bc.curve(P_inter, D_N, k, p_piece, knot)
    # print(P_piece)
    for i in range(piece_num - 1):
        tmp_x = [P_piece[0][i], P_piece[0][i+1]]
        tmp_y = [P_piece[1][i], P_piece[1][i+1]]
        plt.plot(tmp_x, tmp_y, color='g')
    plt.show()

# 曲线拟合法
def curve_approx_figure(D, D_N, k, H):
    '''
    D: trajetory point
    D_N: the length of trajectory
    k: degree
    H: the number of control points
    '''

    '''
    Step 1. Calculate the parameters
    '''
    # p_centripetal = ps.centripetal(D_N, D)
    p_centripetal = ps.uniform_spaced(D_N)
    # print("p_centripetal: ", p_centripetal)

    '''
    Step 2. Calculate the knot vector
    '''
    knot = ps.knot_vector(p_centripetal, k, D_N)
    # print("knot: ", knot)

    '''
    Step 3. Calculate the control points
    '''
    P_control = bc.curve_approximation(D, D_N, H, k, p_centripetal, knot)
    print(P_control)

    fig = plt.figure()
    for i in range(H):
        plt.scatter(P_control[0][i], P_control[1][i], color='b')

    for i in range(D_N):
        plt.scatter(D[0][i], D[1][i], color='r')

    for i in range(H - 1):
        tmp_x = [P_control[0][i], P_control[0][i+1]]
        tmp_y = [P_control[1][i], P_control[1][i+1]]
        plt.plot(tmp_x, tmp_y, color='b')

    '''
    Step 4. Calculate the points on the b-spline curve
    '''
    piece_num = 80
    p_piece = np.linspace(0, 1, piece_num)
    p_centripetal_new = ps.centripetal(H, P_control)
    knot_new = ps.knot_vector(p_centripetal_new, k, H)
    P_piece = bc.curve(P_control, H, k, p_piece, knot_new)

    # print(P_piece)
    for i in range(piece_num - 1):
        tmp_x = [P_piece[0][i], P_piece[0][i+1]]
        tmp_y = [P_piece[1][i], P_piece[1][i+1]]
        plt.plot(tmp_x, tmp_y, color='g')
    plt.show()

    # plt.savefig("./test.png")


def uniformization(tra, len):
    """
    把密度不均匀的轨迹点均匀化（两点之间距离相近）
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
    
        # 如果 i 和 i+1 点的距离小于len，则删除 i+1 点
        if dis > len:
            saveList.append(tra[j, :])
            print("dis = {}".format(dis))
            i = j
    saveList = np.array(saveList)
    print("saveList len: ", saveList.shape)
    # plt.scatter(saveList[:, 0], saveList[:, 1])
    # plt.show()
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

# 曲线拟合
def showTra():
    
    dataDir = '../data/bag_2/'
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]
    
    # d = DouglasPeuker()
    # point = d.main(point)
    # point = np.array(point)
    # print(point.shape)
    # plt.scatter(point[:,0],point[:, 1], color='r')
    # plt.plot(point[:,0],point[:, 1], color='r')
    # point = point.T
    # point[0, :] = point[0, :] - np.average(point[0, :])
    # point[1, :] = 100*(point[1, :] - np.average(point[1, :]))

    point = uniformization(tra , 5)
    # point = reducePoint(point, step=20)

    # d = DouglasPeuker()
    # point = d.main(point)
    # point = np.array(point)

    # d =  LimitVerticalDistance()
    # ddd = d.diluting(point)
    # ddd = np.array(ddd)

    point = point.T
    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])

    # plt.plot(point[0,:],point[1, :], color='r')
    plt.scatter(point[0,:],point[1, :], color='r')
    plt.show()





    k = 3
    H = 15
    D_N = point.shape[1]

    # curve_approx_figure(D=point, D_N=D_N, k=k, H=H)

# 曲线插值
def showTra2():
    dataDir = './data/bag_2/'
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]
    
    # d = DouglasPeuker()
    # point = d.main(point)
    # point = np.array(point)
    # print(point.shape)
    # plt.scatter(point[:,0],point[:, 1], color='r')
    # plt.plot(point[:,0],point[:, 1], color='r')
    # point = point.T
    # point[0, :] = point[0, :] - np.average(point[0, :])
    # point[1, :] = point[1, :] - np.average(point[1, :])

    
    point = reducePoint(point, step=20)
    point = point.T
    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])
    plt.plot(point[0,:],point[1, :], color='r')
    plt.show()

   


    k = 3
    H = 12
    D_N = point.shape[1]
    curve_inter_figure(D=point, D_N=D_N, k=k)
    

# 曲线拟合测试
def showTest():
    D_X = [1, 1, 0, -0.5, 1.5, 3, 4, 4.2, 4]
    D_Y = [0, 1, 2, 3, 4, 3.5, 3, 2.5, 2]
    D = [D_X, D_Y]
    k = 4
    H = 8
    D_N = len(D_X)
    curve_approx_figure(D=D, D_N=D_N, k=k, H=H)

# 曲线插值测试
def showTest2():
    D_X = [1, 1, 0, -0.5, 1.5,   3, 4, 4.2, 4]
    D_Y = [0, 1, 2,    3,   4, 3.5, 3, 2.5, 2]
    D = [D_X, D_Y]
    D_N = len(D_X)
    k = 3 
    curve_inter_figure(D,D_N , k)


if __name__ == '__main__':
    showTra()
    # showTest()
    # showTest2()


