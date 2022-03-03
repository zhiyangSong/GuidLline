import sys

sys.path.append("./pb")
from pb.vehicle_state_pb2 import VehicleState
from pb.localization_pb2 import Pose
from pb.planning_pb2 import State
from pb.planning_pb2 import ADCTrajectory

import numpy as np



filepath = "./dump/1638497966.323/localization_output/localization_output"
pre_str = open(filepath,'rb').read()
# pose  = VehicleState()

pose = Pose()
pose.ParseFromString(pre_str)
x = []
y = []
x.append(pose.position.x) 
y.append(pose.position.y) 
print(x)
print(y)








    



# with open("./dump/1638497966.323/localization_output/localization_output",'rb+') as f:
#     pre_str = f.read()

#     pose  = Pose()
#     pose.ParseFromString(pre_str)

    
#     print(type(pose.position))
    
    