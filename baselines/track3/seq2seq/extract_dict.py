# -*- coding: utf-8 -*-
from transformers import BertTokenizer

model_path = "pretrained-models/bart-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
vocab = tokenizer.get_vocab()
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
with open(f"{model_path}/dict.txt", 'w') as fw:
    for line in vocab_sorted:
        fw.write(f"{line[0]} {line[1]}\n")
