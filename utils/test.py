
import sys
import os
sys.path.append("./pb")

from pb.localization_pb2 import Pose


def getpath():
    for root,dirs,files in os.walk(r"./dump"):
        for file in files:    
            #获取文件路径
            filepath= os.path.join(root,file)
            path.append(filepath)
            # print(filepath)



def getstate():
    for ppath in path:
        pre_str = open(ppath,'rb').read()
        state.ParseFromString(pre_str)
        with open("data.txt","a")as f:
            f.write(str(state.position)+"\n")
        f.close()
        print(state)


if __name__ == '__main__':
    path =[]
    getpath()
    state = Pose()
    getstate()

    

  






