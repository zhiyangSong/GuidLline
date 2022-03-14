
 
from math import sqrt, pow
import numpy as np
import matplotlib.pyplot as plt




'''
垂距限值法其实和DP算法原理一样，但是垂距限值不是从整体角度考虑，而是依次扫描每一个点，检查是否符合要求。

算法过程如下:

1、以第二个点开始，计算第二个点到前一个点和后一个点所在直线的距离d；
2、如果d大于阈值，则保留第二个点，计算第三个点到第二个点和第四个点所在直线的距离d;若d小于阈值则舍弃第二个点，计算第三个点到第一个点和第四个点所在直线的距离d;
3、依次类推，直线曲线上倒数第二个点。
'''




THRESHOLD = 0.07 # 阈值
 
 
def point2LineDistance(point_a, point_b, point_c):
  """  计算点a到点b c所在直线的距离  :param point_a:  :param point_b:  :param point_c:  :return:  """
  # 首先计算b c 所在直线的斜率和截距
  if point_b[0] == point_c[0]:
    return 9999999
  slope = (point_b[1] - point_c[1]) / (point_b[0] - point_c[0])
  intercept = point_b[1] - slope * point_b[0]
 
  # 计算点a到b c所在直线的距离
  distance = abs(slope * point_a[0] - point_a[1] + intercept) / sqrt(1 + pow(slope, 2))
  return distance


 
class LimitVerticalDistance(object):
  def __init__(self):
    self.threshold = THRESHOLD
    self.qualify_list = list()
 
  def diluting(self, point_list):
    """    抽稀    :param point_list:二维点列表    :return:    """
    self.qualify_list.append(point_list[0])
    check_index = 1
    while check_index < len(point_list) - 1:
      distance = point2LineDistance(point_list[check_index],
                     self.qualify_list[-1],
                     point_list[check_index + 1])
 
      if distance < self.threshold:
        check_index += 1
      else:
        self.qualify_list.append(point_list[check_index])
        check_index += 1
    return self.qualify_list
 
 
if __name__ == '__main__':
 

    dataDir = "./data/bag_2/"
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]

    d =  LimitVerticalDistance()
    ddd = d.diluting(point)
    ddd = np.array(ddd)
    print(ddd.shape)
    plt.scatter(ddd[:,0],ddd[:, 1], color='r')
    plt.plot(ddd[:,0],ddd[:, 1], color='r')
    plt.show()
  
 
