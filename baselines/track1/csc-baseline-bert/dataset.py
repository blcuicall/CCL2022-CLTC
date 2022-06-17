import torch
import pickle
from torch.utils.data.dataloader import default_collate
import os


class CSC_Dataset(torch.utils.data.Dataset):

    def __init__(self, path, config, subset):
        self.path = path
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.subset = subset
        self.data = self.read_data()

    def read_data(self):
        data_path = self.path

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # data = data[:10000]

        print("--------------------------------")
        print(self.subset + ": " + str(len(data)))
        print("--------------------------------\n")

        return data

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.LongTensor(self.data[idx]['input_ids'])
        item['token_type_ids'] = torch.LongTensor(self.data[idx]['token_type_ids'])
        item['attention_mask'] = torch.LongTensor(self.data[idx]['attention_mask'])
        if self.subset != 'test':
            item['trg_ids'] = torch.LongTensor(self.data[idx]['trg_ids'])
        return item

    def __len__(self):
        return len(self.data)


class Padding_in_batch:

    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def pad(self, seq, pad_id, max_len):
        pad_part = torch.LongTensor([pad_id] * (max_len - seq.shape[-1]))
        pad_seq = torch.cat([seq, pad_part], dim=-1)
        return pad_seq

    def __call__(self, batch):
        max_len = 0
        for item in batch:
            max_len = max(max_len, len(item["input_ids"]))

        if "trg_ids" in item:
            is_test = False
        else:
            is_test = True

        for item in batch:
            item['input_ids'] = self.pad(item['input_ids'], self.input_pad_id, max_len)
            item['token_type_ids'] = self.pad(item['token_type_ids'], 0, max_len)
            item['attention_mask'] = self.pad(item['attention_mask'], 0, max_len)

            if not is_test:
                item['trg_ids'] = self.pad(item['trg_ids'], self.input_pad_id, max_len)

        return default_collate(batch)
