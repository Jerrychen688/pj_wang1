import torch
from dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join

class Tester:
    def __init__(self, dataset, model, valid_or_test, model_name):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.model_name = model_name
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def get_rank(self, sim_scores):
        # Assumes the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, position):
        r, e1, e2, e3, e4, e5, e6 = fact
        if position == 1:
            return [(r, i, e2, e3, e4, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 2:
            return [(r, e1, i, e3, e4, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 3:
            return [(r, e1, e2, i, e4, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 4:
            return [(r, e1, e2, e3, i, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 5:
            return [(r, e1, e2, e3, e4, i, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 6:
            return [(r, e1, e2, e3, e4, e5, i) for i in range(1, self.dataset.num_ent())]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        return self.shred_facts(result)

    def test(self, test_by_arity=False):
        """
        Evaluate the given dataset and print results, either by arity or all at once
        """
        settings = ["raw", "fil"]
        normalizer = 0
        self.measure_by_arity = {}
        self.meaddsure = Measure()
        current_measure, normalizer =  self.eval_dataset(self.dataset.data[self.valid_or_test])
        self.measure = current_measure
        if normalizer == 0:
            raise Exception("No Samples were evaluated! Check your test or validation data!!")
        self.measure.normalize(normalizer)
        self.measure_by_arity["ALL"] = self.measure
        pr_txt = "Results for ALL ARITIES in {} set".format(self.valid_or_test)
        print(pr_txt)
        print(self.measure)
        return self.measure, self.measure_by_arity

    def eval_dataset(self, dataset):
        """
        Evaluate the dataset with the given model.
        """
        # Reset normalization parameter
        settings = ["raw", "fil"]
        normalizer = 0
        current_rank = Measure()
        for i, fact in enumerate(dataset):
            arity = self.dataset.max_arity - (fact == 0).sum()
            for j in range(1, arity + 1):
                normalizer += 1
                queries = self.create_queries(fact, j)
                for raw_or_fil in settings:
                    r, e1, e2, e3, e4, e5, e6 = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    data = torch.stack([r, e1, e2, e3, e4, e5, e6],dim=1)
                    sim_scores = self.model.forward(data).cpu().data.numpy()
                    rank = self.get_rank(sim_scores)
                    current_rank.update(rank, raw_or_fil)
            if i % 1000 == 0:
                print("--- Testing sample {}".format(i))
        return current_rank, normalizer

    def shred_facts(self, tuples):
        r  = [tuples[i][0] for i in range(len(tuples))]
        e1 = [tuples[i][1] for i in range(len(tuples))]
        e2 = [tuples[i][2] for i in range(len(tuples))]
        e3 = [tuples[i][3] for i in range(len(tuples))]
        e4 = [tuples[i][4] for i in range(len(tuples))]
        e5 = [tuples[i][5] for i in range(len(tuples))]
        e6 = [tuples[i][6] for i in range(len(tuples))]
        return torch.LongTensor(r).to(self.device), torch.LongTensor(e1).to(self.device), torch.LongTensor(e2).to(self.device), torch.LongTensor(e3).to(self.device), torch.LongTensor(e4).to(self.device), torch.LongTensor(e5).to(self.device), torch.LongTensor(e6).to(self.device)

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data.get(spl, ()):
                tuples.append(tuple(fact))
        return tuples



