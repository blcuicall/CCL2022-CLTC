# -*- coding: utf-8 -*-
# Author: Cunliang Kong
# Affiliation: BLCU-ICALL

import argparse
from transformers import BertTokenizer


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--load-dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = options()
    tokenizer = BertTokenizer.from_pretrained(args.load_dir)
    with open(args.input_file) as fr, open(args.output_file, 'w') as fw:
        for line in fr:
            tok_line = tokenizer.tokenize(line.strip())
            fw.write(f"{' '.join(tok_line)}\n")
