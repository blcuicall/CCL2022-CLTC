# -*- coding:UTF-8 -*-
from numpy import isin
from torch.utils.data import Dataset
from utils.helpers import INCORRECT_LABEL, SEQ_DELIMETERS, START_TOKEN, KEEP_LABEL, PAD_LABEL, UNK_LABEL, CORRECT_LABEL
from random import random
import torch
from tqdm import tqdm
import pickle
import re
import os

class Seq2EditVocab:
    def __init__(self, d_vocab_path, c_vocab_path, unk2keep=False):
        self.detect_vocab = self.read_vocab(d_vocab_path)
        self.correct_vocab = self.read_vocab(c_vocab_path, unk2keep)

    def read_vocab(self, path, unk2keep=False):
        id2tag = []
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                id2tag.append(line)
        tag2id = {tag: idx for idx, tag in enumerate(id2tag)}
        unk_label = KEEP_LABEL if unk2keep else UNK_LABEL
        tag2id = Tag2idVocab(tag2id, unk_label)
        return {"id2tag": id2tag, "tag2id": tag2id}

class Tag2idVocab:
    def __init__(self, tag2id: dict, unk_label):
        self.tag2id = tag2id
        self.unk_label = unk_label
    
    def __getitem__(self, key):
        if key in self.tag2id:
            return self.tag2id[key]
        else:
            return self.tag2id[self.unk_label]

class Seq2EditDataset(Dataset):
    def __init__(self, data_path, use_cache, tokenizer, vocab, max_len, tag_strategy, skip_complex=0, skip_correct=0, tp_prob=1, tn_prob=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.tag_strategy = tag_strategy
        self.max_len = max_len
        self.skip_complex = bool(skip_complex)
        self.skip_correct = bool(skip_correct)
        self.tn_prob = tn_prob
        self.tp_prob = tp_prob
        self.vocab = vocab
        if use_cache and os.path.exists(data_path+".pkl"):
            print("Data cache found, we'll load pkl...")
            self.data = self.load_data_from_pkl(data_path+".pkl")
        else:
            
            self.data = self.read_data(data_path)
            if use_cache:
                self.save_data_to_pkl(data_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def read_data(self, path):
        data = list()
        with open(path, "r", encoding="utf8") as fr:
            for line in tqdm(fr):
                line = line.strip()
                if not line:
                    continue
                word_tag_pairs = re.split(f"(?<!{SEQ_DELIMETERS['tokens']}){SEQ_DELIMETERS['tokens']}", line)
                word_tag_pairs = [word_tag.rsplit(SEQ_DELIMETERS["labels"], 1) for word_tag in word_tag_pairs]
                words = [word for word, _ in word_tag_pairs]
                tags = [tag for _, tag in word_tag_pairs]

                if words and words[0] != START_TOKEN:
                    words = [START_TOKEN] + words
                
                if self.max_len is not None:
                    words = words[:self.max_len]
                    tags = tags[:self.max_len]

                input_ids, offsets = self.tokenizer.encode(words)
                instance = self.build_instance(words, input_ids, offsets, tags)
                if instance:
                    data.append(instance["inputs"])
        return data

    def load_data_from_pkl(self, path):
        with open(path, "rb") as fr:
            return pickle.load(fr)

    def save_data_to_pkl(self, path):
        with open(path+".pkl", "wb") as fw:
            fw.write(pickle.dumps(self.data))

    def extract_tags(self, tags):
        correct_tags = [tag_text.split(SEQ_DELIMETERS["operations"]) for tag_text in tags]

        # key是某个word对应的复杂标签的个数，value是这个句子里复杂标签等于这个个数的出现了多少次
        complex_flag_dict = dict()
        for i in range(5):
            idx = i + 1
            complex_flag_dict[idx] = sum([len(x) > idx for x in correct_tags])
        
        if self.tag_strategy == "keep_one":
            correct_tags = [tag[0] for tag in correct_tags]
        elif self.tag_strategy == "merge_all":
            correct_tags = [tag for tag in tags]
        else:
            raise NotImplementedError("Invalid tag strategy! ")
        detect_tags = [CORRECT_LABEL if tag == KEEP_LABEL else INCORRECT_LABEL for tag in correct_tags]
        return detect_tags, correct_tags, complex_flag_dict

    def build_input_dict(self, input_ids, offsets, word_level_len):
        token_type_ids = [0 for _ in range(len(input_ids))]
        attn_mask = [1 for _ in range(len(input_ids))]
        word_mask = [1 for _ in range(word_level_len)]
        # correspond to IndexedTokenList
        input_dict = {
            "input_ids": input_ids, 
            "token_type_ids": token_type_ids, 
            "attention_mask": attn_mask, 
            "offsets": offsets,
            "word_mask": word_mask}
        return input_dict

    def build_instance(self, words, input_ids, offsets, tags):
        instance = dict()
        instance["metadata"] = {"words": words}
        detect_tags, correct_tags, complex_flag_dict = self.extract_tags(tags)
        input_dict = self.build_input_dict(input_ids, offsets, len(words))
        detect_tag_ids = [self.vocab.detect_vocab["tag2id"][tag] for tag in detect_tags]
        correct_tag_ids = [self.vocab.correct_vocab["tag2id"][tag] for tag in correct_tags]

        if self.skip_complex and complex_flag_dict[self.skip_complex] > 0:
            return None

        # skip TN
        rnd = random() # [0.0,1.0)
        if self.skip_correct and all(x == CORRECT_LABEL for x in detect_tags):
            if rnd > self.tn_prob:
                return None
        # skip TP
        else:
            if rnd > self.tp_prob:
                return None
        input_dict["detect_tag_ids"] = detect_tag_ids
        input_dict["correct_tag_ids"] = correct_tag_ids
        instance["inputs"] = input_dict
        return instance


class MyCollate:
    def __init__(self, input_pad_id, detect_pad_id, correct_pad_id):
        self.input_pad_id = input_pad_id
        self.detect_pad_id = detect_pad_id
        self.correct_pad_id = correct_pad_id

    def pad_to_max_len(self, input_seq, max_len, pad_value=0):
        pad_len = max_len - len(input_seq)
        pad_piece = [pad_value for _ in range(pad_len)]
        return input_seq + pad_piece
    
    def pad_instance(self, instance, max_len):
        """
        padding each tensor to max len
        """
        instance["input_ids"] = self.pad_to_max_len(instance["input_ids"], max_len, pad_value=self.input_pad_id)
        instance["token_type_ids"] = self.pad_to_max_len(instance["token_type_ids"], max_len)
        instance["attention_mask"] = self.pad_to_max_len(instance["attention_mask"], max_len)
        instance["word_mask"] = self.pad_to_max_len(instance["word_mask"], max_len)
        
        instance["offsets"] = self.pad_to_max_len(instance["offsets"], max_len, pad_value=(0,0))
        if "detect_tag_ids" in instance:
            instance["detect_tag_ids"] = self.pad_to_max_len(instance["detect_tag_ids"], max_len, pad_value=self.detect_pad_id)
        if "correct_tag_ids" in instance:
            instance["correct_tag_ids"] = self.pad_to_max_len(instance["correct_tag_ids"], max_len, pad_value=self.correct_pad_id)
        return instance

    def __call__(self, batch):

        max_len = max([len(i["input_ids"]) for i in batch])
        
        for item in batch:
            item = self.pad_instance(item, max_len)

        keys = item.keys()

        batch_dict = dict()
        for key in keys:
            value = torch.tensor([item[key] for item in batch], dtype=torch.long)
            batch_dict[key] = value
        return batch_dict
