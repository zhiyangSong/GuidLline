import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from process_data.B_Spline_Approximation import BS_curve  
from process_data.uniformization import uniformization
    
    
"""
test for bspline 两种方法

从包中拿数据,通过b样条处理成 控制点坐标,并拟合成曲线
"""


bs = BS_curve(8,3)

dataDir = './data/bag_20220211_1/'
tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
point = tra[:, :2]
point = uniformization(point , 5)


xx = point[: , 0]
yy = point[: , 1]
plt.scatter(xx ,yy)

paras = bs.estimate_parameters(point)
knots = bs.get_knots()
if bs.check():
    cp = bs.approximation(point)
uq = np.linspace(0,1,101)
y = bs.bs(uq)

plt.plot(y[:,0],y[:,1],'r')
plt.plot(cp[:,0],cp[:,1],'y')
plt.scatter(cp[:,0],cp[:,1],c = 'y')
print("cp.shape:{}".format(cp.shape))
print(cp)
plt.show()





"""
通过控制点和节点出拟合曲线

"""
bs = BS_curve(8,3)
knots = bs.get_knots()
controlPoints = np.array([ 
                [ 0.00000000e+00 , 0.00000000e+00 ], [-1.73422712e+01 , 2.15658488e-01],
                [-2.35382699e+01 , 3.13403809e-01] ,[-3.76766971e+01,  5.46545379e-01],
                [-4.78039077e+01 , 5.62960911e-01 ],[-5.98447978e+01 ,-7.11339258e-02],
                [-7.74681854e+01, -2.17359351e+00] ,[-8.87793826e+01 ,-1.75073904e+00],
                [-9.63270000e+01, -1.81000000e+00]
                            ])
bs.cp = controlPoints
uq = np.linspace(0,1,101)
y = bs.bs(uq)
plt.plot(y[:,0],y[:,1],'r')
plt.show()