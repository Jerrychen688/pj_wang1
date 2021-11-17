import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math
import dataset
from attention import GATLayer
from convolution import HGNN_conv

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)  # 权值是不断学习的所以要是parameter类型
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

class HGCN(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(HGCN, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)  # 实体个数，嵌入维数，索引指定填充
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs["hidden_drop"]
        self.decay = kwargs["decay"]
        self.alpha = kwargs["alpha"]
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate) # torch.nn.Dropout(0.5), 这里的 0.5 是指该层（layer）的神经元在每次迭代训练时会随机有 50% 的可能性被丢弃（失活），不参与训练，一般多神经元的 layer 设置随机失活的可能性比神经元少的高。
        self.att=GATLayer(dataset.num_ent(),dataset.num_rel())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset: dataset.Dataset = dataset
        self.max_arity = 6
        self.traingraph1,self.traingraph2= self.dataset.getSparseGraph(self.dataset.data["train"], device=self.device)
        #self.hgc1 = HGNN_conv(self.emb_dim, self.emb_dim)         the parameter about HyperGCN


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])

    def att_graph(self,data):
        att_matrix ,rmb= self.att.forward(self.E.weight, self.R.weight)
        imp, unimp = self.att.cal(data, att_matrix)
        self.graph1, self.graph2= self.dataset.getSparseGraph(imp, device=self.device)
        return imp,unimp

    '''
    covolutional network
    because this prt involve later work,the code can not be released for the time being
    '''
    #def computer(self, Graph):


    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)  # 左右拼接,把两个张量拼接在一起，二维张量第0维取全部，第一维取sh到最后
        return y

    def forward(self, minibatch):
        r_idx, e1_idx, e2_idx, e3_idx, e4_idx,e5_idx,e6_idx = self.dataset.each(minibatch, device=self.device)
        r = self.R(r_idx)
        e_emb = self.computer(self.graph1) #four-layer concolutional
        e1 = e_emb[e1_idx]
        e2 = self.shift(e_emb[e2_idx], int(1 * self.emb_dim / self.max_arity))
        e3 = self.shift(e_emb[e3_idx], int(2 * self.emb_dim / self.max_arity))
        e4 = self.shift(e_emb[e4_idx], int(3 * self.emb_dim / self.max_arity))
        e5 = self.shift(e_emb[e5_idx], int(4 * self.emb_dim / self.max_arity))
        e6 = self.shift(e_emb[e6_idx], int(5 * self.emb_dim / self.max_arity))
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        loss = torch.norm(self.E(e1_idx), p=2)+torch.norm(self.E(e2_idx), p=2)+torch.norm(self.E(e3_idx), p=2)+torch.norm(self.E(e4_idx), p=2)
        reg_loss = loss * self.decay
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x,reg_loss

    def forward1(self, S1):
        r_idx, e1_idx, e2_idx, e3_idx, e4_idx,e5_idx,e6_idx = self.dataset.each(S1, device=self.device)
        matrix,rmb=self.att(self.E.weight.data,self.R.weight.data)
        r = rmb[r_idx]
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        loss = torch.norm(self.R(r_idx), p=2)+torch.norm(self.E.weight,p=2)
        reg_loss = loss * self.decay
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x,reg_loss




