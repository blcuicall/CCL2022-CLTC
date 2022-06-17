# -*- coding:UTF-8 -*-
from src.trainer import Trainer
from argparse import ArgumentParser
import deepspeed
import os

def main(args):
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", help="path to deepspeed config json", default="./deepspeed_config.json")
    parser.add_argument("--local_rank", default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_pieces_per_token", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--accumulation_size", type=int, default=1)
    parser.add_argument("--valid_batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--tp_prob", type=float, default=1)
    parser.add_argument("--tn_prob", type=float, default=1)
    parser.add_argument("--additional_confidence", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cold_lr", type=float, default=1e-3)
    parser.add_argument("--dp_rate", type=float, default=0.0)
    parser.add_argument("--cold_step_count", type=int, default=0)
    parser.add_argument("--sub_token_mode", type=str, default="average")
    parser.add_argument("--tag_strategy", type=str, default="keep_one")
    parser.add_argument("--unk2keep", type=int, default=0, help="replace oov label with keep")
    parser.add_argument("--special_tokens_fix", type=int, default=0)
    parser.add_argument("--skip_complex", type=int, default=0)
    parser.add_argument("--skip_correct", type=int, default=0)
    parser.add_argument("--detect_vocab_path", type=str, required=True)
    parser.add_argument("--correct_vocab_path", type=str, required=True)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=False)
    parser.add_argument("--use_cache", default=1, type=int, help="use processed data cache")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--pretrained_transformer_path", type=str, required=True)
    args = parser.parse_args()
    parser = deepspeed.add_config_arguments(parser)
    main(args)
