'''
道格拉斯-普克(Douglas-Peuker)算法

Douglas-Peuker算法(DP算法)过程如下:

1、连接曲线首尾两点A、B；
2、依次计算曲线上所有点到A、B两点所在曲线的距离；
3、计算最大距离D，如果D小于阈值threshold,则去掉曲线上出A、B外的所有点；如果D大于阈值threshold,则把曲线以最大距离分割成两段；
4、对所有曲线分段重复1-3步骤，知道所有D均小于阈值。即完成抽稀。
这种算法的抽稀精度与阈值有很大关系，阈值越大，简化程度越大，点减少的越多；反之简化程度越低，点保留的越多，形状也越趋于原曲线。
'''

from math import sqrt, pow
import numpy as np
import matplotlib.pyplot as plt
 
THRESHOLD = 0.05 # 阈值
 
 
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
 
 
class DouglasPeuker(object):
  def __init__(self):
    self.threshold = THRESHOLD
    self.qualify_list = list()
    self.disqualify_list = list()
 
  def diluting(self, point_list):
    """    抽稀    :param point_list:二维点列表    :return:    """
    if len(point_list) < 3:
      self.qualify_list.extend(point_list[::-1])
    else:
      # 找到与首尾两点连线距离最大的点
      max_distance_index, max_distance = 0, 0
      for index, point in enumerate(point_list):
        if index in [0, len(point_list) - 1]:
          continue
        distance = point2LineDistance(point, point_list[0], point_list[-1])
        if distance > max_distance:
          max_distance_index = index
          max_distance = distance
 
      # 若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割
      if max_distance < self.threshold:
        self.qualify_list.append(point_list[-1])
        self.qualify_list.append(point_list[0])
      else:
        # 将曲线按最大距离的点分割成两段
        sequence_a = point_list[:max_distance_index]
        sequence_b = point_list[max_distance_index:]
 
        for sequence in [sequence_a, sequence_b]:
          if len(sequence) < 3 and sequence == sequence_b:
            self.qualify_list.extend(sequence[::-1])
          else:
            self.disqualify_list.append(sequence)
            
            
 


  def main(self, point_list):
    self.diluting(point_list)
    while len(self.disqualify_list) > 0:
        self.diluting(self.disqualify_list.pop())
    print(self.qualify_list)
    print(len(self.qualify_list))
    return self.qualify_list
 

if __name__ == '__main__':
    d = DouglasPeuker()
    dataDir = "./data/bag_play_base/"
    tra = np.loadtxt("{}tra.csv".format(dataDir), delimiter=",", dtype="double")
    point = tra[:, :2]
    ddd = d.main(point)
    ddd = np.array(ddd)
    print(ddd.shape)
    plt.scatter(ddd[:,0],ddd[:, 1], color='r')
    plt.plot(ddd[:,0],ddd[:, 1], color='r')
    plt.show()
 