
import sys
import numpy as np

import os
sys.path.append("./pb")

from pb.localization_pb2 import Pose

# 文件取出的顺序变
def getpath(): 
    for root,dirs,files in os.walk(r"./dump"): 
        for file in files:    
            #获取文件路径
                filepath= os.path.join(root,file)
                path_this.append(filepath)
                print(filepath)
                
                with open("path.txt","a" ) as f:
                    f.write(filepath+"\n")
                f.close()


# 可以按照文件夹顺序拿到
def getpath2():
    path ="./dump"    #指定需要读取文件的目录
    files =os.listdir(path) #采用listdir来读取所有文件
    files.sort() #排序
               
    for file_ in files:     #循环读取每个文件名
        filepath= os.path.join(path,file_)
        # print(filepath)
        filepath = filepath + "/localization_output/localization_output"
        path_this.append(filepath)
        # print(filepath)
        with open("path.txt","a" ) as f:
                f.write(filepath+"\n")
        f.close()
        

# 获取位置点坐标
def getstate():
    x = []
    y = []
    for ppath in path_this:
        pre_str = open(ppath,'rb').read()
        state.ParseFromString(pre_str)
       
        x.append(state.position.x) 
        y.append(state.position.y)    
        with open("data_pos.txt","a")as f:
            f.write(str(state.position))
        f.close()

        with open("data_vel.txt","a" ) as f:
             f.write(str(state.velocity))
        f.close()
       
        print(state.position)
    return x,y

if __name__ == '__main__':
    path_this=[]
    
    getpath2()
    state = Pose()
    x ,y = getstate()
    x = np.array(x)
    y = np.array(y)
    point = np.vstack([x,y])
    print(point[0, :])
    np.save('./point.npz',point)
    # print(x)
    

    

  






