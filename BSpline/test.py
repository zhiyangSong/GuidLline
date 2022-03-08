
import numpy as np
import b_spline as bs
import matplotlib.pyplot as plt




point = np.load("./position.npy")
print(point)
D_X = point[0]
D_Y = point[1,:]
# point = bs.get_control_point(D_X , D_Y)
# print(point)
# np.save('./control_point' ,point )

# plt.plot(D_X  ,D_Y)
# plt.show()
# print(D_X)


bs.curve_approx_figure(D_X , D_Y)


