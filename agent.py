import numpy as np
import os, sys
import random
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from net import BCNet


class BCAgent():
    def __init__(self, input_size, output_size, args):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.device = args.device
        self.args = args
        
        self.features, self.labels = self.getData()
        
        
        self.net = BCNet(self.input_size, self.output_size, args)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
    
    def getData(self):
        features = None
        labels = None
        return features, labels


    def data_iter(self, batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, batch_size):
            j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
            yield  features.index_select(0, j), labels.index_select(0, j)

    def train(self):
        # https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
        for epoch in range(self.args.num_epochs):
            for X, y in self.data_iter(self.batch_size, self.features, self.labels): 
                pred = self.net(X)
                loss = self.loss_function(pred, y)

                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();
            
            if epoch%self.args.save_interval == 0:
                self.save(epoch)


    def save(self,episodes):
        time_now = time.strftime('%y%m_%d%H%M')
        if not os.path.exists(dir):
            print("dont have this dir")
            os.mkdir(dir)
        dir = 'model/{}_{}episodes.pth'.format(time_now,episodes)
        torch.save(self.net.state_dict(), dir)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        print('load network successed')



