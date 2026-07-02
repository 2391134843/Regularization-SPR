# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim
import json
from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, HGE_TNTComplEx, HGE_TComplEx
from models_TLTs import HGE_TLT_KGE
from regularizers import N3, Lambda3
from itertools import product
import time
import os

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'HGE_TNTComplEx', 'HGE_TComplEx',
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=10, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)

parser.add_argument(
    '--device', default="cpu", help="CPU or cuda device"
)
parser.add_argument(
    '--attention', default=0, type=int,
    help="use temporal-relational attention or not"
)
parser.add_argument(
    '--num_form', default="split", help="which num form will the model take"
)

parser.add_argument('--cycle', type=int, default=1, help='Cycle parameter')  # Add this line


args = parser.parse_args()


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


train_mrr_values = []
train_hits1_values = []
train_hits3_values = []
train_hits10_values = []
valid_mrr_values = []
valid_hits1_values = []
valid_hits3_values = []
valid_hits10_values = []
test_mrr_values = []
test_hits1_values = []
test_hits3_values = []
test_hits10_values = []

def run(args):
    dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()
    model = {
        'ComplEx': ComplEx(sizes, args.rank),
        'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'HGE_TNTComplEx': HGE_TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb,num_form=args.num_form,use_attention=args.attention),
        'HGE_TComplEx': HGE_TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb,num_form=args.num_form,use_attention=args.attention),
        'HGE_TLT_KGE': HGE_TLT_KGE(sizes, args.rank, cycle=args.cycle, num_form=args.num_form, use_reverse=False,
                                   attention=0)
    }[args.model]
    model = model.to(args.device)
    best_hits1 = 0
    best_res_test = {}
    save_path = "expe_log/{}_{}_{}_{}_{}_{}_{}_{}".format(args.dataset, args.model, args.rank, args.learning_rate,
                                                          args.emb_reg, args.time_reg, args.attention,args.num_form)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    emb_reg = N3(args.emb_reg)
    time_reg = Lambda3(args.time_reg)

    for epoch in range(args.max_epochs):
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()
        if dataset.has_intervals():
            optimizer = IKBCOptimizer(
                model, emb_reg, time_reg, opt, dataset,
                batch_size=args.batch_size
            )
            optimizer.epoch(examples)

        else:
            optimizer = TKBCOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=args.batch_size
            )
            optimizer.epoch(examples)

        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            if dataset.has_intervals():
                train, test, valid = [
                    dataset.eval(model, split, -1 if split != 'train' else 10000)
                    for split in ['train', 'test', 'valid']
                ]
                print("valid: ", valid)
                print("test: ", test)
                print("train: ", train)
                print("epoch: ", epoch)
                if valid['hits@_all'][0] > best_hits1:
                    torch.save(
                        {'result': test, 'param': model.state_dict()},
                        '{}/best.pth'.format(save_path, args.model, args.dataset))
                    print('best')
                    best_hits1 = valid['hits@_all'][0]
                    best_res_test = test
            else:
                train, test, valid = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['train', 'test', 'valid']
                ]
                # print("valid: ", valid['MRR'],"valid_Hits: ", valid['hits@[1,3,10]'])
                # print("test_MRR: ", test['MRR'])
                # print("test_Hits: ", test['hits@[1,3,10]'])
                # print("train_mrr: ", train['MRR'],"train_Hits: ", train['hits@[1,3,10]'])
                # print("epoch: ", epoch)
                # Extract the metrics for each split

                train_mrr = train['MRR']
                train_hits1, train_hits3, train_hits10 = train['hits@[1,3,10]']
                valid_mrr = valid['MRR']
                valid_hits1, valid_hits3, valid_hits10 = valid['hits@[1,3,10]']
                test_mrr = test['MRR']
                test_hits1, test_hits3, test_hits10 = test['hits@[1,3,10]']

                # Store the metrics in the corresponding lists
                train_mrr_values.append(train_mrr)
                train_hits1_values.append(train_hits1)
                train_hits3_values.append(train_hits3)
                train_hits10_values.append(train_hits10)
                valid_mrr_values.append(valid_mrr)
                valid_hits1_values.append(valid_hits1)
                valid_hits3_values.append(valid_hits3)
                valid_hits10_values.append(valid_hits10)
                test_mrr_values.append(test_mrr)
                test_hits1_values.append(test_hits1)
                test_hits3_values.append(test_hits3)
                test_hits10_values.append(test_hits10)


                # Print the metrics
                print(f"Epoch {epoch+1}:")
                print(f"  Train MRR: {train_mrr}, Train Hits@1: {train_hits1}, Train Hits@3: {train_hits3}, Train Hits@10: {train_hits10}")
                print(f"  Valid MRR: {valid_mrr}, Valid Hits@1: {valid_hits1}, Valid Hits@3: {valid_hits3}, Valid Hits@10: {valid_hits10}")
                print(f"  Test MRR: {test_mrr}, Test Hits@1: {test_hits1}, Test Hits@3: {test_hits3}, Test Hits@10: {test_hits10}")
                
                          

                # ---------------------------------
                if valid['hits@[1,3,10]'][0] > best_hits1:
                    torch.save(
                        {'MRR': test['MRR'], 'Hist': test['hits@[1,3,10]'], 'param': model.state_dict(),'args':args},
                        '{}/best.pth'.format(save_path, args.model, args.dataset))
                    print('best')
                    best_hits1 = valid['hits@[1,3,10]'][0]
                    best_res_test = test
    # Optionally, save the metrics to a file at the end of each epoch (or at the end of training)
    with open("training_metrics.txt", "a") as f:
        # f.write(f"Epoch {epoch}:\n")
        f.write(f"  Train MRR: {train_mrr_values}\n Train Hits@1: {train_hits1_values}\n ")
        f.write(f"Train Hits@3: {train_hits3_values}\n Train Hits@10: {train_hits10_values}\n")
        f.write(f"  Valid MRR: {valid_mrr_values}\n, Valid Hits@1: {valid_hits1_values}\n")
        f.write(f"Valid Hits@3: {valid_hits3_values}\n Valid Hits@10: {valid_hits10_values}\n")
        f.write(f"  Test MRR: {test_mrr_values}\n Test Hits@1: {test_hits1_values}\n ")
        f.write(f"Test Hits@3: {test_hits3_values}\n Test Hits@10: {test_hits10_values}\n") 
    return best_res_test

test_result = run(args)
print(test_result)


