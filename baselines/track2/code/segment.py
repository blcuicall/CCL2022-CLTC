# coding:utf-8


import sys
import tokenization
from tqdm import tqdm


tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt", do_lower_case=True)

for line in tqdm(sys.stdin):
    line = line.strip()
    items = line.split('\t')
    # items = "", line

    line = tokenization.convert_to_unicode(items[1])
    if not line:
        print()
        continue

    tokens = tokenizer.tokenize(line)
    print(' '.join(tokens))

