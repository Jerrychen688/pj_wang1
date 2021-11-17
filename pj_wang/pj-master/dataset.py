import os
import numpy as np
import random
import torch
import math

class Dataset:
    def __init__(self, ds_name, max_arity=6):
        self.name = ds_name
        self.dir = os.path.join("data", ds_name)
        self.max_arity = max_arity
        # id zero means no entity. Entity ids start from 1.
        self.ent2id = {"":0}
        self.rel2id = {"":0}
        self.data = {}
        print("Loading the dataset {} ....".format(ds_name))
        self.data["train"] = self.read(os.path.join(self.dir, "train.txt"))
        np.random.shuffle(self.data['train'])

        # Load the test data
        self.data["test"] = self.read(os.path.join(self.dir, "test.txt"))
        # Read the test files by arity, if they exist
        # If they do, then test output will be displayed by arity
        for i in range(2,self.max_arity+1):
            test_arity = "test_{}".format(i)
            file_path = os.path.join(self.dir, "test_{}.txt".format(i))
            self.data[test_arity] = self.read_test(file_path)
        self.data["valid"] = self.read(os.path.join(self.dir, "valid.txt"))
        self.batch_index = 0

    def read(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = np.zeros((len(lines), self.max_arity + 1))
        for i, line in enumerate(lines):
            tuples[i] = self.tuple2ids(line.strip().split("\t"))
        return tuples

    def read_test(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = np.zeros((len(lines),  self.max_arity + 1))
        for i, line in enumerate(lines):
            splitted = line.strip().split("\t")[1:]
            tuples[i] = self.tuple2ids(splitted)
        return tuples

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def tuple2ids(self, tuple_):
        output = np.zeros(self.max_arity + 1)
        for ind,t in enumerate(tuple_):
            if ind == 0:
                output[ind] = self.get_rel_id(t)
            else:
                output[ind] = self.get_ent_id(t)
        return output

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    def rand_ent_except(self, ent):
        # id 0 is reserved for nothing. randint should return something between zero to len of entities
        rand_ent = random.randint(1, self.num_ent() - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def next_batch(self, attbatch,neg_ratio):
        pos_batch = self.next_pos_batch(attbatch)
        batch = self.generate_neg(pos_batch, neg_ratio)
        labels = batch[:, 7]
        return batch, labels

    def next_pos_batch(self, batch):
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the +1 label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the 0 arity
        return batch

    def generate_neg(self, pos_batch, neg_ratio):
        arities = [8 - (t == 0).sum() for t in pos_batch]
        pos_batch[:, -1] = arities
        neg_batch = np.concatenate([self.neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], neg_ratio, c) for i, c in enumerate(pos_batch)], axis=0)
        return neg_batch

    def next_pos_batch1(self, attbatch,batch_size):
        if self.batch_index + batch_size < len(attbatch):
            batch = attbatch[self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = attbatch[self.batch_index:]
            ###shuffle##
            np.random.shuffle(attbatch)
            self.batch_index = 0
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the +1 label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the 0 arity
        return batch

    def generate_neg1(self, pos_batch, neg_ratio):
        pos_batch=self.next_pos_batch(pos_batch)
        arities = [8 - (t == 0).sum() for t in pos_batch]
        pos_batch[:,-1] = arities
        neg_batch = np.concatenate([self.neg_each(np.repeat([c], neg_ratio * arities[i] + 1, axis=0), arities[i], neg_ratio,c) for i, c in enumerate(pos_batch)], axis=0)
        labels = neg_batch[:, 7]
        return neg_batch,labels

    def neg_each(self, arr, arity, nr, c):
        arr[0, -2] = 1
        for a in range(arity):
            for j in range(6):
                if c[j + 1] != 0 and a == 0:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
                    break
                elif c[j + 1] != 0 and a == 1:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
                    break
                elif c[j + 1] != 0 and a == 2:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
                    break
                elif c[j + 1] != 0 and a == 3:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
                    break
                elif c[j + 1] != 0 and a == 4:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
                    break
                elif c[j + 1] != 0 and a == 5:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
                    break
                elif c[j + 1] != 0 and a == 6:
                    arr[a * nr + 1:(a + 1) * nr + 1, j + 1] = np.random.randint(low=1, high=self.num_ent(), size=nr)
                    c[j + 1] = 0
        return arr

    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))

    def each(self,batch,device):
        r  = torch.tensor(batch[:,0]).long().to(device)
        e1 = torch.tensor(batch[:,1]).long().to(device)
        e2 = torch.tensor(batch[:,2]).long().to(device)
        e3 = torch.tensor(batch[:,3]).long().to(device)
        e4 = torch.tensor(batch[:,4]).long().to(device)
        e5 = torch.tensor(batch[:,5]).long().to(device)
        e6 = torch.tensor(batch[:,6]).long().to(device)
        return r, e1, e2, e3, e4, e5, e6

    def getSparseGraph(self, data, device):
        H1 = torch.zeros((len(self.rel2id), len(self.ent2id)))
        H2 = torch.zeros((len(self.rel2id), len(self.ent2id)))
        a = torch.tensor(data[:, 0]).long().to(device)
        b = torch.tensor(data[:, 1]).long().to(device)
        c = torch.tensor(data[:, 2]).long().to(device)
        d = torch.tensor(data[:, 3]).long().to(device)
        f = torch.tensor(data[:, 4]).long().to(device)
        g = torch.tensor(data[:, 5]).long().to(device)
        h = torch.tensor(data[:, 6]).long().to(device)
        l = torch.stack([b, c, d, f,g,h], dim=0)
        for j in l:
            for i, m in zip(a, j):
                if m == 0:
                    continue
                elif H2[i, m] >= 1:
                    H2[i, m] += 1
                else:
                    H1[i, m] = 1
                    H2[i, m] = 1
        H11 = H1.T
        B1 = torch.sum(H1, dim=1)
        B11 = list(B1)
        W1 = torch.zeros((len(self.rel2id), len(self.rel2id)))
        for i in range(0, len(self.rel2id)):
            W1[i, i] = ((B11[i] - (min(B11) + 1)) / (max(B11) - (min(
                B11) + 1)))  # W[i, i] = ((B1[i] - B1[minB1]) / (B1[maxB1] - B1[minB1]))  # W[i, i] = ((B1[i] - (minB1+1)) / (maxB1 - (minB1+1)))
        W1[0, 0] = 0.
        B1[B1 == 0.] = 1
        WB1 = W1 / B1
        D1 = torch.sum(torch.mm(W1, H1), dim=0)
        H1 = torch.tensor(H1, dtype=torch.float)
        WB1 = torch.tensor(WB1, dtype=torch.float)
        H11 = torch.tensor(H11, dtype=torch.float)
        WH1 = torch.spmm(WB1, H1)
        HWH1 = torch.spmm(H11, WH1)
        D1[D1 == 0.] = 1.
        D1_sqrt = torch.sqrt(D1).unsqueeze(dim=0)
        HWH1 = HWH1 / D1_sqrt
        Graph1 = HWH1 / D1_sqrt.t()

        H22 = H2.T
        B2 = torch.sum(H2, dim=1)
        B22 = list(B2)
        W2 = torch.zeros((len(self.rel2id), len(self.rel2id)))
        for i in range(0, len(self.rel2id)):
            W2[i, i] = ((B22[i] - (min(B22) + 1)) / (max(B22) - (min(
                B22) + 1)))
        W2[0, 0] = 0.
        B2[B2 == 0.] = 1
        WB2 = W2 / B2
        D2 = torch.sum(torch.mm(W2, H2), dim=0)
        H2 = torch.tensor(H2, dtype=torch.float)
        WB2 = torch.tensor(WB2, dtype=torch.float)
        H22 = torch.tensor(H22, dtype=torch.float)
        WBH2 = torch.spmm(WB2, H2)
        HWBH2 = torch.spmm(H22, WBH2)
        D2[D2 == 0.] = 1.
        D2_sqrt = torch.sqrt(D2).unsqueeze(dim=0)
        HWBH2 = HWBH2 / D2_sqrt
        Graph2 = HWBH2 / D2_sqrt.t()
        return Graph1, Graph2

