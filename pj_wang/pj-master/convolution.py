import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_ft, out_ft),requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        support = torch.mm(input1, self.weight)
        output = torch.spmm(input2, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
