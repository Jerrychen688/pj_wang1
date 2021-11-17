import sys
import threading
import warnings
from world import cprint, cprint1
warnings.filterwarnings(action='ignore')
import os
import glob
import re
import json
import datetime
import numpy as np
import torch
from torch.nn import functional as F
from models import *
import argparse
from dataset import Dataset
from tester import Tester
import math
DEFAULT_SAVE_DIR = 'output'
DEFAULT_MAX_ARITY = 6

class Experiment:
    def __init__(self, args):
        self.model_name = args.model
        self.learning_rate = args.lr
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.neg_ratio = args.nr
        self.max_arity = DEFAULT_MAX_ARITY
        self.pretrained = args.pretrained
        self.test = args.test
        self.output_dir = args.output_dir
        self.restartable = args.restartable
        self.decay = args.decay
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 定义device，其中需要注意的是“cuda:0”代表起始的device_id为0，如果直接是“cuda”，同样默认是从0开始。可以根据实际需要修改起始位置，如“cuda:1”。
        self.kwargs = {"hidden_drop":args.hidden_drop,"decay":args.decay,"alpha":args.alpha}
        self.hyperpars = {"model":args.model,"lr":args.lr,"emb_dim":args.emb_dim,"nr":args.nr, "hidden_drop":args.hidden_drop}
        self.dataset = Dataset(args.dataset, DEFAULT_MAX_ARITY)
        self.num_iterations = args.num_iterations
        # Create an output dir unless one is given
        self.output_dir = self.create_output_dir(args.output_dir)
        self.measure = None
        self.measure_by_arity = None
        self.test_by_arity = not args.no_test_by_arity
        self.best_model = None
        self.load_model()
        self.save_hparams(args)
        self.minibatch, self.targets = self.dataset.next_batch(self.dataset.data['train'], neg_ratio=self.neg_ratio)

    def decompose_predictions(self, targets, predictions, max_length): #
        positive_indices = np.where(targets > 0)[0]
        seq = []
        for ind, val in enumerate(positive_indices):
            if(ind == len(positive_indices)-1):
                seq.append(self.padd(predictions[val:], max_length))
            else:
                seq.append(self.padd(predictions[val:positive_indices[ind + 1]], max_length))
        return seq

    def padd(self, a, max_length):
        b = F.pad(a, (0,max_length - len(a)), 'constant', -math.inf)
        return b

    def padd_and_decompose(self, targets, predictions, max_length):
        seq = self.decompose_predictions(targets, predictions, max_length)
        return torch.stack(seq)

    def load_last_saved_model(self, output_dir):
        model_found = False
        try:
            model_list = [os.path.basename(x) for x in glob.glob(os.path.join(self.output_dir, 'model_*.chkpnt'))]
            model_list_sorted = sorted(model_list, key=lambda f: int(re.match(r'model_(\d+)itr.chkpnt', f).groups(0)[0]))
        except:
            print("*** NO SAVED MODEL FOUND in {}. LOADING FROM SCRATCH. ****".format(self.output_dir))
            self.model.init()
        else:
            if len(model_list_sorted) > 0:
                self.pretrained = os.path.join(self.output_dir, model_list_sorted[-1])
                opt_path = os.path.join(os.path.dirname(self.pretrained), os.path.basename(self.pretrained).replace('model','opt'))
                if os.path.exists(opt_path):
                    self.model.load_state_dict(torch.load(self.pretrained))
                    self.opt.load_state_dict(torch.load(opt_path))
                    model_found = True
                    try:
                        best_model_path = os.path.join(self.output_dir, "best_model.chkpnt")
                        self.best_model = self.get_model_from_name(self.model_name)
                        self.best_model.load_state_dict(torch.load(best_model_path))
                        print("Loading the model {} with best MRR {}.".format(self.pretrained, self.best_model.best_mrr))
                    except:
                        print("*** NO BEST MODEL FOUND in {}. ****".format(self.output_dir))
                        self.best_model = None
        if not model_found:
            print("*** NO MODEL/OPTIMIZER FOUND in {}. LOADING FROM SCRATCH. ****".format(self.output_dir))
            self.model.init()

    def load_model(self):
        print("Initializing the model ...")
        self.model = HGCN(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.pretrained is not None:
            print("Loading the pretrained model at {} for testing".format(self.pretrained))
            self.model.load_state_dict(torch.load(self.pretrained))
            opt_path = os.path.join(os.path.dirname(self.pretrained), os.path.basename(self.pretrained).replace('model','opt'))
            if os.path.exists(opt_path):
                self.opt.load_state_dict(torch.load(opt_path))
            elif not self.test:
                raise Exception("*** NO OPTIMIZER FOUND. SKIPPING. ****")
        elif self.restartable and os.path.isdir(self.output_dir):
            self.load_last_saved_model(self.output_dir)
        else:
            self.model.init()

    def test_and_eval(self):
        print("Testing the {} model on {}...".format(self.model_name, self.dataset.name))
        self.model.eval()
        with torch.no_grad():
            tester = Tester(self.dataset, self.model, "test", self.model_name)
            self.measure, self.measure_by_arity = tester.test(self.test_by_arity)
            self.save_model(self.model.cur_itr, "test")

    def train_and_eval(self):
        if (self.model.cur_itr.data >= self.num_iterations):
            print("*************")
            print("Number of iterations is the same as that in the pretrained model.")
            print("Nothing left to train. Exiting.")
            print("*************")
            return
        print("Training the {} model...".format(self.model_name))
        print("Number of training data points: {}".format(len(self.dataset.data["train"])))
        loss_layer = torch.nn.CrossEntropyLoss()
        print("Starting training at iteration ... {}".format(self.model.cur_itr.data))
        for it in range(self.model.cur_itr.data, self.num_iterations + 1):
            last_batch = False
            self.model.train()
            self.model.cur_itr.data += 1
            attlosses = 0
            covlosses = 0
            self.opt.zero_grad()
            attscore,regolss = self.model.forward1(self.minibatch)
            targets = torch.FloatTensor(self.targets)
            attloss = F.binary_cross_entropy_with_logits(attscore, targets)
            attloss=regolss+attloss
            attloss.backward()
            self.opt.step()
            attlosses += attloss.item()
            while not last_batch:
                minibatch1 = self.dataset.next_pos_batch1(self.dataset.data['train'], self.batch_size)
                last_batch = self.dataset.was_last_batch()
                attbatch, unattbatch = self.model.att_graph(minibatch1)
                minibatch2, targets2 = self.dataset.generate_neg1(attbatch, neg_ratio=self.neg_ratio)
                predictions, loss = self.model.forward(minibatch2)
                predictions = self.padd_and_decompose(targets2, predictions,self.neg_ratio * self.max_arity)  # 把正样本的评分值取出来
                number_of_positive = len(np.where(targets2 > 0)[0])
                targets2 = torch.zeros(number_of_positive).long().to(self.device)
                covloss = loss_layer(predictions, targets2)
                regloss1 = (loss / number_of_positive) * self.decay
                covloss = covloss + regloss1
                self.opt.zero_grad()
                covloss.backward()
                self.opt.step()
                covlosses += covloss.item()
            print("Iteration#: {},loss: {},{}".format(it,attlosses,covlosses))
            if (it % 100 == 0) or (it == self.num_iterations):
                self.model.eval()
                with torch.no_grad():
                    print("validation:")
                    tester = Tester(self.dataset, self.model, "valid", self.model_name)
                    measure_valid, _ = tester.test()
                    mrr = measure_valid.mrr["fil"]
                    is_best_model = (self.best_model is None) or (mrr > self.best_model.best_mrr)
                    if is_best_model:
                        self.best_model = self.model
                        self.best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                        self.best_model.best_itr.data = torch.from_numpy(np.array([it]))
                    self.save_model(it, "valid", is_best_model=is_best_model)
        self.best_model.eval()
        with torch.no_grad():
            cprint("testing best model at iteration {} .... ".format(self.best_model.best_itr))
            tester = Tester(self.dataset, self.best_model, "test", self.model_name)
            self.measure, self.measure_by_arity = tester.test(self.test_by_arity)
        print("Saving model at {}".format(self.output_dir))
        self.save_model(it, "test")

    def create_output_dir(self, output_dir=None):
        if output_dir is None:
            time = datetime.datetime.now()
            model_name = '{}_{}_{}'.format(self.model_name, self.dataset.name, time.strftime("%Y%m%d-%H%M%S"))
            output_dir = os.path.join(DEFAULT_SAVE_DIR, self.model_name, model_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created output directory {}".format(output_dir))
        return output_dir

    def save_model(self, itr=0, test_or_valid='test', is_best_model=False):
            if is_best_model:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.chkpnt'))
                print("######## Saving the BEST MODEL")
                params = self.model.state_dict()
            model_name = 'model_{}itr.chkpnt'.format(itr)
            opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(self.model_name)
            measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)
            print("######## Saving the model {}".format(os.path.join(self.output_dir, model_name)))
            print()
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name))
            torch.save(self.opt.state_dict(), os.path.join(self.output_dir, opt_name))
            if self.measure is not None:
                measure_dict = vars(self.measure)
                if self.best_model:
                    measure_dict["best_iteration"] = self.best_model.best_itr.cpu().item()
                    measure_dict["best_mrr"] = self.best_model.best_mrr.cpu().item()
                with open(os.path.join(self.output_dir, measure_name), 'w') as f:
                        json.dump(measure_dict, f, indent=4, sort_keys=True)
            if (self.test_by_arity) and (self.measure_by_arity is not None):
                H = {}
                measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)
                for key in self.measure_by_arity:
                    H[key] = vars(self.measure_by_arity[key])
                with open(os.path.join(self.output_dir, measure_by_arity_name), 'w') as f:
                        json.dump(H, f, indent=4, sort_keys=True)

    def save_hparams(self, args):
        args_dict = vars(args)
        hparams_name = "hparams_test.json" if args.test else "hparams.json"
        with open(os.path.join(self.output_dir, hparams_name), 'w') as f:
            json.dump(args_dict, f, indent=4, sort_keys=True)

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HGCN")
    parser.add_argument('-dataset', type=str, default="FB-AUTO")
    parser.add_argument('-lr', type=float, default=0.003)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-alpha', type=float, default=0.3)
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-num_iterations', type=int, default=1200)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('-decay', type=float, default=0.001,help="the weight decay for l2 normalizaton")
    parser.add_argument("-test", action="store_true", help="If -test is set, then you must specify a -pretrained model. "
                        + "This will perform testing on the pretrained model and save the output in -output_dir")
    parser.add_argument("-no_test_by_arity", action="store_true", help="If set, then validation will be performed by arity.")
    parser.add_argument('-pretrained', type=str, default=None, help="A path to a trained model (.chkpnt file), which will be loaded if provided.")
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the model will be saved and/or loaded from.")
    parser.add_argument('-restartable', action="store_true", help="If restartable is set, you must specify an output_dir")
    args = parser.parse_args()
    if args.restartable and (args.output_dir is None):
            parser.error("-restarable requires -output_dir.")
    experiment = Experiment(args)
    if args.test:
        print("************** START OF TESTING ********************", experiment.model_name)
        if args.pretrained is None:
            raise Exception("You must provide a trained model to test!")
        experiment.test_and_eval()
    else:
        print("************** START OF TRAINING ********************", experiment.model_name)
        experiment.train_and_eval()
