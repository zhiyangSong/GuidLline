import os
from matplotlib import pyplot as plt 
from process import *
import numpy as np
import glob
from process_data.B_Spline_Approximation import BS_curve
from process_data.uniformization import uniformization






def autoMkdir(first_dir, last_dir):
    """
    根据bag包文件名创建相应的文件夹
    first_dir: bag 包文件路径
    last_dir: 需要创建的文件夹路径
    """
    for root, dirs, files in os.walk(first_dir):
        for file in files:
            file = file.split('.b')[0]
            desDir = "{}/{}".format(last_dir, file)
            if not os.path.exists(desDir):
                os.mkdir(desDir)
    
def fun2(dataDir):
    """ 删除特定文件每条数据中的道路文件 """
    fileDirs = glob.glob(pathname = '{}/bag_2022*'.format(dataDir))
    for file in fileDirs:
        seg_file_list = glob.glob(pathname='{}/segment*.csv'.format(file))
        for seg_file in seg_file_list:
            try:
                os.remove(seg_file)
            except:
                print("删除文件%s异常" % seg_file)

def fun():
    """
    把各个路段数据整合
    """
    fea_1 = np.load('./data_input/features_1.npy')
    fea_2 = np.load('./data_input/features_2.npy')
    features = np.vstack([fea_1, fea_2])
    print(features.shape)
    np.save('./data_input/features', features)

    lab_1 = np.load('./data_input/labels_1.npy')
    lab_2 = np.load('./data_input/labels_2.npy')
    labels = np.vstack([lab_1, lab_2])
    print(labels.shape)
    np.save('./data_input/labels', labels)





limitConfig = {
    "data_1": [-200, -100, 0],      # x 轴坐标
    "data_2": [-3910, -3810, 1]     # y 轴坐标
}


if __name__ == '__main__':
    dataDir = './data'
    traDir = './data/bag_20220108_1'
    juncDir = './data/junction'
    limit = limitConfig['data_1']
    index = 1   # 区分生成的数据
    # 打印路段信息
    # plotMap(traDir=traDir, juncDir=juncDir, segBegin=0, segEnd=0)

    # preProcess(juncDir=juncDir)    # 路段数据预处理：根据计算边界
    pltTra(dataDir=dataDir, juncDir=juncDir, traDir=traDir)     # 打印轨迹

    # ############
    # 数据一条数据
    # fea, lab = getTrainData(traDir=traDir, juncDir=juncDir, limit_1=limit[0], limit_2=limit[1], axis=limit[2])
    # 对路段内所有数据进行处理
    # fea, lab = batchProcess(dataDir=dataDir, juncDir=juncDir, limit=limit, index=index)
    # print("fea shape: ", fea.shape, " lab shape: ", lab.shape)


    # augmentData(juncDir=juncDir, traDir=traDir, angle=np.pi/2, show=True)