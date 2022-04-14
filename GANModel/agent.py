from cProfile import label
import numpy as np
import os, sys
import random
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, tensor
from torch.utils.tensorboard import SummaryWriter
from net import Discriminator ,Generator


class GANAgent():
    def __init__(self, input_size, output_size, args):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.args = args
        
        self.features, self.labels = self.getData()
        
        self.Gnet = Generator(self.input_size, self.output_size, args)
        self.Dnet = Discriminator(self.output_size, args)

        self.optimizer_G = torch.optim.Adam(self.Gnet.parameters(), lr=self.learning_rate)
        self.optimizer_D = torch.optim.Adam(self.Dnet.parameters(), lr=self.learning_rate)

        self.loss_function = nn.MSELoss(size_average=False)     # reduction='sum'

        time_now = time.strftime('%y%m_%d%H%M')
        self.save_dir = "{}/{}".format(args.save_dir, time_now)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.log_dir = "./{}/{}".format(args.log_dir,time_now)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)


    
    def getData(self):
        features = np.load(self.args.fea_dir)
        labels = np.load(self.args.lab_dir)
        return features, labels


    def data_iter(self, batch_size, features, labels):
        """
        随机返回一个 batch_size 的数据
        """
        num_examples = features.shape[0]
        indices = np.random.permutation(num_examples)
        for i in range(0, num_examples, batch_size):

            j = indices[i: min(i + batch_size, num_examples)]
            if j.shape[0]<batch_size: break
            
            yield features[j, :], labels[j, :]




    def gradient_penalty(self , D ,xr, xf):
        """

        :param D:
        :param xr:
        :param xf:
        :return:
        """

        LAMBDA = 0.3
        # only constrait for Discriminator
        xf = xf.detach()
        xr = xr.detach()
        #[b,1]
        t = torch.rand(xr.shape[0] , 1)
        t = t.expand_as(xr)
        # print("after:t:{}".format(t.shape))
        # print("xr:{}".format(xr.shape))
        mid = t * xr +(1-t) * xf
        # print(mid.shape)
        #set it requires gradient
        mid.requires_grad_()

        pred = D(mid)
        grads = autograd.grad(outputs=pred, inputs=mid,
                                grad_outputs=torch.ones_like(pred),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gp



    def train(self):
        writer = SummaryWriter(self.log_dir)

        for epoch in range(self.args.episodes_num):

            #1 train Discriminator firstly
            for _ in range(5):
                 #1 train on real data


                for X, y in self.data_iter(self.batch_size, self.features, self.labels): 
                    
                    #1 train on real data
                   
                    yr = torch.FloatTensor(y)
                    predr = self.Dnet(yr)
                    lossr = -predr.mean()
                    #1.2 train on fake data
                    z  = torch.randn(self.batch_size,self.input_size)
                    yf = self.Gnet(z).detach()
                    predf = self.Dnet(yf)
                    lossf = predf.mean()
                    #1.3 gradient penalty
                    gp  = self.gradient_penalty(self.Dnet,yr, yf)

                     #aggregate all
                    loss_D = lossr+ lossf+gp
                    # loss_D = lossr+ lossf

                    #optimize
                    self.optimizer_D.zero_grad()
                    loss_D.backward()
                    self.optimizer_D.step()
            

            # 2 train Generator
            z = torch.randn(self.batch_size , self.input_size)
            yf = self.Gnet(z)
            predf = self.Dnet(yf)
            #max predf.mean()
            loss_G = -predf.mean()
           
            #optimize
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()


            
            writer.add_scalar("train_lossD", loss_D.item(), epoch)
            writer.add_scalar("train_lossG", loss_G.item(), epoch)
                # if epoch%self.args.save_interval == 0:
                #     self.save(epoch)
            print(f"epoch: {epoch:<4} loss_G: {loss_G:>7f}")
            print(f"epoch: {epoch:<4} loss_D: {loss_D:>7f}")

        self.save(epoch)
        writer.close()
     


    def save(self, episodes):
        dir = '{}/episodes_{}.pth'.format(self.save_dir,episodes)
        torch.save(self.Gnet.state_dict(), dir)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        print('load network successed')



