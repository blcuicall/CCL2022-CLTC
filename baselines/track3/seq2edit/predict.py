# -*- coding:UTF-8 -*-
from src.predictor import Predictor
import re
from argparse import ArgumentParser
from tqdm import tqdm
import time
import deepspeed

def read_batch(path, batch_size, segmented):
    batch = []
    with open(path, "r", encoding="utf8") as fr:
        while True:
            try:
                line = next(fr)
            except StopIteration:
                yield batch
                break
            line = line.strip()
            if segmented:
                line = line.split(" ")
            else:
                line = list(line)
            if len(batch) < batch_size:
                batch.append(line)
            else:
                yield batch
                batch = [line]

def detokenize(text: str):
    text = re.sub(" ##(?=\S)", "", text)
    text = re.sub("\s+", "", text)
    # text = re.sub("(?<![a-zA-Z]) | (?![a-zA-Z])", "", text)
    return text

def main(args):
    predictor = Predictor(args)
    total_corrections = []
    cnt_corrections = 0
    print(f"model path: {args.model_dir}")
    print(f"ckpt id: {args.ckpt_id}")
    print("start predicting ...")
    s = time.time()
    if args.out_path:
        fw = open(args.out_path, "w", encoding="utf8")
    for batch_text in tqdm(read_batch(args.input_path, args.batch_size, args.segmented)):
        pred_batch, cnt = predictor.handle_batch(batch_text)
        cnt_corrections += cnt
        if args.out_path:
            for idx, pred_tokens in enumerate(pred_batch):
                pred_line = " ".join(pred_tokens)
                if not re.search("[^ #]", pred_line):
                    pred_line = " ".join(batch_text[idx])
                    print("prediction for current line is none, thus we replace it with source line...")
                    print(args.ckpt_id)
                    print(pred_line)
                if bool(args.detokenize):
                    pred_line = detokenize(pred_line)
                fw.write(pred_line+"\n")
        else:
            for pred_tokens in pred_batch:
                pred_line = " ".join(pred_tokens)
                if bool(args.detokenize):
                    pred_line = detokenize(pred_line)
                print(pred_line)
    e = time.time()
    if args.out_path:
        fw.close()
    print(f"total cost: {e -s }s")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--iteration_count", type=int, default=5)
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_error_probability", type=float, default=0.0)
    parser.add_argument("--additional_confidence", type=float, default=0.0)
    parser.add_argument("--sub_token_mode", type=str, default="average")
    parser.add_argument("--max_pieces_per_token", type=int, default=5)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ckpt_id", type=str, required=True)
    parser.add_argument("--detect_vocab_path", type=str, required=True)
    parser.add_argument("--correct_vocab_path", type=str, required=True)
    parser.add_argument("--pretrained_transformer_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--special_tokens_fix", type=int, default=0)
    parser.add_argument("--segmented", type=int, default=0)
    parser.add_argument("--detokenize", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(args)
