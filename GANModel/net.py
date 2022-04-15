import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self,input_size ,output_size ,args):
        super(Generator ,self).__init__()
        self.net = nn.Sequential(
            # z:[b,2]   =>[b,2]
            nn.Linear(input_size,args.num_units_1),
            nn.ReLU(True),
            nn.Linear(args.num_units_1 ,  args.num_units_2),
            nn.ReLU(True),
            nn.Linear(args.num_units_2 , output_size),
           
        )

    def forward(self ,z):
        output = self.net(z)
        return output



class Discriminator(nn.Module):

    def __init__(self,output_size ,args):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(output_size, args.num_units_1),
            nn.ReLU(True),
            nn.Linear(args.num_units_1, args.num_units_2),
            nn.ReLU(True),
            nn.Linear(args.num_units_2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)