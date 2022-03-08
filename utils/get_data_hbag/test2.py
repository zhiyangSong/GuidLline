import sys
import os
sys.path.append("./pb")
from pb.vehicle_state_pb2 import VehicleState
from pb.localization_pb2 import Pose
from pb.sl_boundary_pb2 import SLBoundary
from pb.planning_pb2 import ADCTrajectory


def getpath():
    for root,dirs,files in os.walk(r"./dump"):
        
        for file in files:    
            #获取文件路径
            filepath= os.path.join(root,file)
            path.append(filepath)

def getstate():
    for ppath in path:
    # print(ppath)
        pre_str = open(ppath,'rb').read()
        state.ParseFromString(pre_str)
        with open("data_planning.txt","a")as f:
            f.write(str(state.trajectory_point)+"\n")
        f.close()
        print(state)



if __name__ == '__main__':
    path =[]
    getpath()
    state = ADCTrajectory()
    # state = SLBoundary()
    getstate()
  



