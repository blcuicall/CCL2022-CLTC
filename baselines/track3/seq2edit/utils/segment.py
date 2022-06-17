# coding:utf-8
import re
import tokenization
from tqdm import tqdm
from argparse import ArgumentParser

def read_line(path):
    with open(path, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            yield line

def ssplit(text):
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    return text

def main(args):
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab, do_lower_case=args.lowercase)
    for doc in tqdm(read_line(args.input)):
        if args.sent_split:
            lines = ssplit(doc).split("\n")
        else:
            lines = [doc]
        for line in lines:
            line = line.strip()

            line = tokenization.convert_to_unicode(line)
            if not line:
                print()
                continue

            tokens = tokenizer.tokenize(line)
            print(' '.join(tokens))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="input file", required=True)
    parser.add_argument("--vocab", type=str, help="vocab file", required=True)
    parser.add_argument("--sent_split", action="store_true", help="whether to split sent for Chinese")
    parser.add_argument("--lowercase", action="store_true", help="whether to lowercase tokens")
    args = parser.parse_args()
    main(args)

