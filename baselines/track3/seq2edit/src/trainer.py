# -*- coding:UTF-8 -*-
import torch
from transformers import AutoTokenizer
from utils.mismatched_utils import *
from utils.common_utils import init_dataloader, read_config, torch_distributed_master_process_first
from src.dataset import Seq2EditVocab
from utils.helpers import INCORRECT_LABEL, KEEP_LABEL, PAD_LABEL, START_TOKEN
from src.model import GECToRModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from random import seed
import os
import json
import deepspeed

class Trainer:
    def __init__(self, args):
        self.use_amp = args.amp
        self.fix_seed()
        self.device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"use device: {self.device}")
        deepspeed.init_distributed()
        self.num_epochs = args.num_epochs
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.do_eval = args.do_eval
        self.lr = args.lr
        self.cold_lr = args.cold_lr
        self.cold_step_count = args.cold_step_count
        self.accumulation_size = args.accumulation_size
        self.max_len = args.max_len
        self.max_pieces_per_token = args.max_pieces_per_token
        self.tp_prob= args.tp_prob
        self.tn_prob = args.tn_prob
        self.tag_strategy= args.tag_strategy
        self.skip_complex= bool(args.skip_complex)
        self.skip_correct= bool(args.skip_correct)
        self.train_path = args.train_path
        self.valid_path = args.valid_path
        self.use_cache = bool(args.use_cache)
        self.model_dir = args.model_dir
        self.ckpt_id = args.ckpt_id
        self.save_dir = args.save_dir
        self.vocab = Seq2EditVocab(args.detect_vocab_path, args.correct_vocab_path, unk2keep=bool(args.unk2keep))
        self.base_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_transformer_path, use_fast=False)
        if bool(args.special_tokens_fix): # for roberta
            self.base_tokenizer.add_tokens([START_TOKEN], special_tokens=True)
            self.base_tokenizer.vocab[START_TOKEN] = self.base_tokenizer.unk_token_id
        self.mismatched_tokenizer = MisMatchedTokenizer(self.base_tokenizer, self.max_len, self.max_pieces_per_token)
        
        model = GECToRModel(
            encoder_path=args.pretrained_transformer_path,
            num_detect_tags=len(self.vocab.detect_vocab["id2tag"]),
            num_correct_tags=len(self.vocab.correct_vocab["id2tag"]),
            additional_confidence=args.additional_confidence,
            dp_rate = args.dp_rate,
            detect_pad_id = self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id= self.vocab.correct_vocab["tag2id"][PAD_LABEL],
            detect_incorrect_id=self.vocab.detect_vocab["tag2id"][INCORRECT_LABEL],
            correct_keep_id=self.vocab.correct_vocab["tag2id"][KEEP_LABEL],
            sub_token_mode=args.sub_token_mode,
            device = self.device
        )

        self.train_loader = init_dataloader(
            subset="train",
            data_path=self.train_path,
            use_cache=self.use_cache,
            tokenizer=self.mismatched_tokenizer,
            vocab=self.vocab,
            input_pad_id=self.base_tokenizer.pad_token_id,
            detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL],
            max_len=self.max_len,
            batch_size=int(self.train_batch_size / torch.distributed.get_world_size()),
            tag_strategy=self.tag_strategy,
            skip_complex=self.skip_complex,
            skip_correct=self.skip_correct,
            tp_prob=self.tp_prob,
            tn_prob=self.tn_prob)
        print("train set: ", len(self.train_loader.dataset))

        self.valid_loader = None
        if args.do_eval:
            self.valid_loader = init_dataloader(
                subset="train",
                data_path=self.valid_path,
                use_cache=self.use_cache,
                tokenizer=self.mismatched_tokenizer,
                vocab=self.vocab,
                input_pad_id=self.base_tokenizer.pad_token_id,
                detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
                correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL],
                max_len=self.max_len,
                batch_size=int(self.valid_batch_size / torch.distributed.get_world_size()),
                tag_strategy=self.tag_strategy,
                skip_complex=self.skip_complex,
                skip_correct=self.skip_correct,
                tp_prob=self.tp_prob,
                tn_prob=self.tn_prob)
            print("dev set: ", len(self.valid_loader.dataset))

        config = read_config(args.config_path)
        # total_batch_size = batch_size_per_gpu * accum_size * num_gpus
        config["train_batch_size"] = self.train_batch_size * self.accumulation_size
        config["gradient_accumulation_steps"] = args.accumulation_size
        config["optimizer"]["params"]["lr"] = args.lr
        config["amp"]["enabled"] = self.use_amp

        self.scheduler = None
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(model=model,
                                        model_parameters=model.parameters(),
                                        config=config)
        if self.model_dir and self.ckpt_id:
            self.model.load_checkpoint(self.model_dir, self.ckpt_id)
            print(f"load model from {self.model_dir}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        else:
            print("no model checkpoint found, train from beginning...")

        self.best_accuracy = 0
        self.best_epoch = 0
        self.best_loss = float("inf")

    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.encoder_requires_grad = True
        
        for epoch in range(self.num_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            if self.cold_step_count:
                
                if epoch < self.cold_step_count:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.cold_lr
                    self.encoder_requires_grad = False
                else:
                    if self.encoder_requires_grad == False:
                        if self.use_amp:
                            print("clean autocast cache...")
                            torch.clear_autocast_cache()
                        torch.cuda.empty_cache()
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr
                        self.encoder_requires_grad = True
                       

            train_loss = self._train_epoch()
            if self.do_eval:
                self.model.eval()

                valid_loss, valid_acc = self._valid_epoch()
                with torch_distributed_master_process_first(torch.distributed.get_rank()):                
                    metrics = self.eval_model(epoch, train_loss, valid_loss, valid_acc)
                if torch.distributed.get_rank() == 0:
                    print(metrics)

            self._save_ckpt(epoch)

    def _save_ckpt(self, epoch):
        self.model.save_checkpoint(self.save_dir, f"epoch-{epoch}")

    def _save_metric(self, epoch, metrics):
        with open(os.path.join(self.save_dir, f"metrics_epoch-{epoch}.json"), "w", encoding="utf8") as fw:
            fw.write(json.dumps(metrics, ensure_ascii=False, indent=2))

    def is_overflow(self):
        if hasattr(self.optimizer, "overflow"):
            return self.optimizer.overflow

    def _train_epoch(self):
        epoch_loss = 0
        pbar = tqdm(self.train_loader)
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()
            outputs = self.model(batch, self.encoder_requires_grad)
            loss = outputs["loss"]
            self.model.backward(loss)
            # call optimizer.step() and lr_scheduler.step()
            self.model.step()
            loss_i = loss.detach().item()
            pbar.set_postfix({'loss': loss_i})
            epoch_loss += loss_i   
        epoch_loss /= len(self.train_loader)
        return epoch_loss

    def eval_model(self, epoch, train_loss, valid_loss, valid_acc):
        metric_path = os.path.join(self.save_dir, f"metrics_epoch-{epoch}.json")
        if os.path.exists(metric_path):
            with open(metric_path, "r", encoding="utf8") as fr:
                metrics = json.load(fr)
        else:
            metrics = {"current_epoch": epoch, "train_loss": train_loss, \
                "valid_loss": valid_loss, "valid_accuracy": valid_acc}
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
            if valid_acc > self.best_accuracy:
                self.best_accuracy = valid_acc
                self.best_epoch = epoch
            metrics["best_epoch"] = self.best_epoch
            metrics["best_valid_loss"] = self.best_loss
            metrics["best_valid_accuracy"] = self.best_accuracy
            self._save_metric(epoch, metrics)
        return metrics

    def _valid_epoch(self):
        
        epoch_loss = 0
        all_pred_labels = list()
        all_gold_labels = list()
        with torch.no_grad():
            for batch in tqdm(self.valid_loader):
                for k, v in batch.items():
                    batch[k] = v.cuda()
                outputs = self.model(batch)
                 
                loss = outputs["loss"]
                loss = loss / torch.distributed.get_world_size()
                epoch_loss += loss.detach().item()
                batch_word_mask = batch["word_mask"].cpu().bool()
                batch_pred_label_probs = outputs["class_probabilities_labels"].detach().cpu()
                batch_pred_labels = torch.argmax(batch_pred_label_probs, dim=-1)
                batch_pred_labels = torch.masked_select(batch_pred_labels, batch_word_mask).tolist()
                all_pred_labels.extend(batch_pred_labels)
                batch_gold_labels = torch.masked_select(batch["correct_tag_ids"].cpu(), batch_word_mask).tolist()
                all_gold_labels.extend(batch_gold_labels)
            epoch_loss /= len(self.valid_loader)
            acc = accuracy_score(all_gold_labels, all_pred_labels)
        return epoch_loss, acc

    def fix_seed(self):
        torch.manual_seed(1)
        if not self.use_amp:
            torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(43)
