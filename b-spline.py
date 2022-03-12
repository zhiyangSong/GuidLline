import BSpline.parameter_selection as ps
import numpy as np
import BSpline.bspline_curve as bc
import matplotlib.pyplot as plt



'''
通过给出的一些轨迹点，反求控制点，画出B样条曲线
'''
def curve_inter_figure():
    '''
    Input: Data points
    '''
    D_X = [1, 1, 0, -0.5, 1, 3, 4, 4.2, 4]
    D_Y = [0, 1, 2,    3, 1, 1, 3, 2.5, 2]
    D = [D_X, D_Y]
    D_N = len(D_X)
    k = 2               # degree

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
    # print(P_inter)

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
    p_centripetal = ps.centripetal(D_N, D)
    # p_centripetal = ps.uniform_spaced(D_N)
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
    # print("P_control", P_control)

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

def showTra():
    dataDir = './data/bag_2/'
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]
    point = reducePoint(point, step=7)
    point = point.T

    point[0, :] = point[0, :] - np.average(point[0, :])
    point[1, :] = point[1, :] - np.average(point[1, :])

    k = 5
    H = 8
    D_N = point.shape[1]

    curve_approx_figure(D=point, D_N=D_N, k=k, H=H)

def showTest():
    D_X = [1, 1, 0, -0.5, 1.5, 3, 4, 4.2, 4]
    D_Y = [0, 1, 2, 3, 4, 3.5, 3, 2.5, 2]
    D = [D_X, D_Y]
    k = 4
    H = 8
    D_N = len(D_X)
    curve_approx_figure(D=D, D_N=D_N, k=k, H=H)

if __name__ == '__main__':
    
    curve_inter_figure()


    