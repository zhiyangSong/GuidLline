from process import plotMap, plotLane, getTrainData

import glob

dataDir = "./data/bag_2/"
path_file_number=glob.glob(pathname='{}segment*.csv'.format(dataDir))
print(path_file_number)
# plotMap(dataDir, segBegin=0, segEnd=0)

# plotLane(dataDir)

getTrainData(dataDir, limit_1=-200, limit_2=-100)