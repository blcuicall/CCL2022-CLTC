# -*- coding:UTF-8 -*-
from argparse import ArgumentParser
from urllib import request
import os

def read_vocab(path):
    assert os.path.exists(path)
    with open(path, "r", encoding="utf8") as fr:
        vocab = {word.strip() for word in fr.read().strip().split("\n") if word.strip()}
    return vocab

def main(args):
    counter = 0
    vocab = read_vocab(args.vocab)
    with open(args.output, "w", encoding="utf8") as fw:
        fw.write("$KEEP\n$DELETE\n")
        for word in vocab:
            fw.write(f"$REPLACE_{word}\n")
            fw.write(f"$APPEND_{word}\n")
            counter += 2
        fw.write("@@UNKNOWN@@\n@@PADDING@@\n")
    counter += 4
    print(f"total labels: {counter}")

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vocab", help="vocab file", required=True)
    parser.add_argument("--output", help="output path", required=True)
    args = parser.parse_args()
    main(args)