
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from process_data.B_Spline_Approximation import BS_curve
from process_data.uniformization import uniformization
    
    
"""
test for bspline
"""


bs = BS_curve(8,3)

dataDir = './data/bag_20220211_1/'
tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
point = tra[:, :2]
point = uniformization(tra , 5)


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