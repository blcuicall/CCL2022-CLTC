import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable
from torch.utils.data import Dataset


class data_loader(Dataset):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, flag, args, batch_size, test=False, cuda=True, src_flag=True,
                 hyp_flag=True):
        self.cuda = cuda
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.data_path = data_path
        self.flag = flag
        self.test = test
        examples = self.read_file()
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
        self.step = 0

    def read_file(self):
        examples = list()
        with open(self.data_path) as fin:
            for line in fin:
                line = line.strip()
                data = json.loads(line)
                src = data["src"]
                if self.flag == "train":
                    for hyp in data["hyps"]:
                        examples.append([src, hyp["text"], hyp["f05"]])
                        break
                elif self.flag == "test":
                    for hyp in data["hyps"]:
                        examples.append([src, hyp["text"]])
                else:
                    raise Exception('flag error')
        return examples

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collect_fn(self, examples):
        if self.flag == "train":
            batch_label = list()
            batch_data = list()
            for example in examples:
                batch_data.append((" ".join(example[0]), example[1]))
                batch_label.append(example[2])
            batch_data = self.tokenizer.batch_encode_plus(
                batch_data,
                max_length=self.max_len,
                padding='longest',
                return_tensors='pt',
                truncation=True
            )
            batch_label = Variable(
                torch.LongTensor(batch_label))
            batch_data["labels"] = batch_label


        elif self.flag == "test":
            batch_data = list()
            for example in examples:
                batch_data.append((" ".join(example[0]), example[1]))
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
