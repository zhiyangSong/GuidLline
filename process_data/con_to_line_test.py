import numpy as np
import matplotlib.pyplot as plt
import sys

from B_Spline_Approximation import BS_curve
from uniformization import uniformization

from controltoline import main1


#通过控制点求轨迹的方法

#先用B_Spline_Approximation.py的方法求出一个knots
bs = BS_curve(8,3)

dataDir = '../data/bag_20220111_3/'
tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
point = tra[:, :2]
point = uniformization(tra , 5)

# 输入的数据点
xx = point[: , 0]
yy = point[: , 1]


paras = bs.estimate_parameters(point)
knots = bs.get_knots()


#通过控制点和节点坐标将曲线画出来
controlPoints = np.array([ 
                [ 0.00000000e+00 , 0.00000000e+00 ], [-1.73422712e+01 , 2.15658488e-01],
                [-2.35382699e+01 , 3.13403809e-01] ,[-3.76766971e+01,  5.46545379e-01],
                [-4.78039077e+01 , 5.62960911e-01 ],[-5.98447978e+01 ,-7.11339258e-02],
                [-7.74681854e+01, -2.17359351e+00] ,[-8.87793826e+01 ,-1.75073904e+00],
                [-9.63270000e+01, -1.81000000e+00]
                            ])
main1(controlPoints,knots)
