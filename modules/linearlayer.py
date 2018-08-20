import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias = True):
        super(Linear,self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias = bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self,x):
        return self.linear(x)
