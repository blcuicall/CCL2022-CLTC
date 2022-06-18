import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import (
    AutoTokenizer,
    AutoModel
)
from torch.utils.data import DataLoader, SequentialSampler
from data_loader import data_loader
from models import inference_model
from data_loader import data_loader
from bert_model import BertForSequenceEncoder
from torch.nn import NLLLoss
import logging
import json

logger = logging.getLogger(__name__)
# 设置种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def eval_model(model, args):
    examples = list()
    with open(args.test_path ) as fhyp: #+ ".hyp"
        for line in fhyp:
            examples.append(line.strip())
    # total_step = int(np.ceil(len(examples) * 1.0 / args.batch_size))
    model.eval()
    predicts = list()
    with torch.no_grad():
        # for step  in range(total_step):
        for step, batch_data in enumerate(geneset_reader):
            inp_tensor = batch_data["input_ids"].cuda()
            msk_tensor = batch_data["attention_mask"].cuda()
            seg_tensor = batch_data["token_type_ids"].cuda()
            prob = model(inp_tensor, msk_tensor, seg_tensor)
            prob = prob[:, 1]
            prob = prob.view(-1).tolist()
            predicts.extend(prob)

    with open(args.out_path, "w")  as fout:
        for predict in predicts:
            fout.write(str(predict) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--out_path', help='output path')
    parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--flag', type=str, default="generate", help='choose the way for dataloader')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--seed", type=int, default=4321,
                        help="random seed for initialization")
    args = parser.parse_args()

    # 随机数种子
    set_seed(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    gene_dataset = data_loader(
        args.test_path, tokenizer, args.flag, args, batch_size=args.batch_size, test=True)
    sampler = SequentialSampler(gene_dataset)

    geneset_reader = DataLoader(
        gene_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=gene_dataset.collect_fn)
    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    logger.info('Start eval!')
    eval_model(model, args)
