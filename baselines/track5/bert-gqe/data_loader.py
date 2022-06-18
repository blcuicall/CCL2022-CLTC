import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable
from torch.utils.data import Dataset


class data_loader(Dataset):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, flag, args, batch_size, test=False, cuda=True):
        self.cuda = cuda
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.data_path = data_path
        self.test = test
        self.flag = flag
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            # self.shuffle()
        self.step = 0

    def read_file(self, data_path):
        examples = list()
        with open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                if self.flag == "train":
                    sublines = line.strip().split("\t")
                    examples.append([sublines[0], int(sublines[1])])
                elif self.flag == "generate":
                    line = line.strip()
                    data = json.load(line)
                    src = data["src"]
                    for hyp in data["hyps"]:
                        examples.append([src, hyp["text"]])
                    # examples.append(line.strip())
                else:
                    raise Exception('flag error')
        return examples

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.examples)

    # 用于迭代数据
    # 通过index获取数据
    def __getitem__(self, index):
        return self.examples[index]

    def collect_fn(self, examples):
        # 传入一个batch的数据
        if self.flag == "train":
            batch_data = list()
            batch_label = list()
            tokens = list()
            for example in examples:
                tokens.append(example[0])
                batch_label.append(example[1])
            batch_data = self.tokenizer.batch_encode_plus(
                tokens,
                max_length=self.max_len,
                padding='longest',
                return_tensors='pt',
                truncation=True

            )
            # if self.test!=True:
            batch_label = Variable(
                torch.LongTensor(batch_label))
            batch_data["labels"] = batch_label

        elif self.flag == "generate":
            batch_data = list()
            for example in examples:
                batch_data.append(example)
            batch_data = self.tokenizer.batch_encode_plus(
                batch_data,
                max_length=self.max_len,
                padding='longest',
                return_tensors='pt',
                truncation=True

            )
        else:

            raise Exception('flag error')
        return batch_data
