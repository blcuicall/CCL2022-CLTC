# -*- coding:UTF-8 -*-
import re
from transformers import AutoTokenizer
from utils.mismatched_utils import *
from src.dataset import Seq2EditVocab, MyCollate
from utils.helpers import INCORRECT_LABEL, KEEP_LABEL, PAD_LABEL, START_TOKEN, UNK_LABEL, get_target_sent_by_edits
from src.model import GECToRModel
from random import seed
import deepspeed


class Predictor:
    def __init__(self, args):
        self.use_amp = args.amp
        print(f"amp: {self.use_amp}")
        self.fix_seed()
        deepspeed.init_distributed()
        self.device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.iteration_count = args.iteration_count
        self.min_len = args.min_len
        self.max_len = args.max_len
        self.min_error_probability = args.min_error_probability
        self.max_pieces_per_token = args.max_pieces_per_token
        self.vocab = Seq2EditVocab(args.detect_vocab_path, args.correct_vocab_path)
        self.base_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_transformer_path, use_fast=False)
        if bool(args.special_tokens_fix): # for roberta
            self.base_tokenizer.add_tokens([START_TOKEN], special_tokens=True)
            self.base_tokenizer.vocab[START_TOKEN] = self.base_tokenizer.unk_token_id
        self.mismatched_tokenizer = MisMatchedTokenizer(self.base_tokenizer, self.max_len, self.max_pieces_per_token)
        self.collate_fn = MyCollate(
            input_pad_id=self.base_tokenizer.pad_token_id,
            detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL])
        self.model = self.init_model(args)
        self.model.eval()

    def init_model(self, args):
        model = GECToRModel(
            encoder_path=args.pretrained_transformer_path,
            num_detect_tags=len(self.vocab.detect_vocab["id2tag"]),
            num_correct_tags=len(self.vocab.correct_vocab["id2tag"]),
            additional_confidence=args.additional_confidence,
            dp_rate = 0.0,
            detect_pad_id = self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id= self.vocab.correct_vocab["tag2id"][PAD_LABEL],
            detect_incorrect_id=self.vocab.detect_vocab["tag2id"][INCORRECT_LABEL],
            correct_keep_id=self.vocab.correct_vocab["tag2id"][KEEP_LABEL],
            sub_token_mode=args.sub_token_mode,
            device = self.device
        )
        ds_engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())
        ds_engine.load_checkpoint(args.model_dir, args.ckpt_id)

        return ds_engine



    def handle_batch(self, full_batch):
        final_batch = full_batch[:]
        # {sent idx: sent}, used for stop iter early
        prev_preds_dict = {idx: [sent] for idx, sent in enumerate(final_batch)}
        short_skip_id_set = set([idx for idx, sent in enumerate(final_batch) if len(sent) < self.min_len])
        # idxs for len(sent) > min_len
        pred_ids = [idx for idx in range(len(full_batch)) if idx not in short_skip_id_set]
        total_updates = 0

        for n_iter in range(self.iteration_count):
            ori_batch = [final_batch[i] for i in pred_ids]
            batch_input_dict = self.preprocess(ori_batch)
            if not batch_input_dict:
                break
            label_probs, label_ids, max_detect_incor_probs = self.predict(batch_input_dict)
            del batch_input_dict
            # list of sents(each sent is a list of target tokens)
            pred_batch = self.postprocess(ori_batch, label_probs, label_ids, max_detect_incor_probs)

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)
            total_updates += cnt
            if not pred_ids:
                break
        return final_batch, total_updates

    def predict(self, batch_inputs):
        with torch.no_grad():
            for k, v in batch_inputs.items():
                batch_inputs[k] = v.cuda()
            outputs = self.model(batch_inputs)
        label_probs, label_ids = torch.max(outputs['class_probabilities_labels'], dim=-1)
        max_detect_incor_probs = outputs['max_error_probability']
        return label_probs.tolist(), label_ids.tolist(), max_detect_incor_probs.tolist()
    
    def preprocess(self, seqs):
        seq_lens = [len(seq) for seq in seqs if seq]
        if not seq_lens:
            return []
        # +1 for [START_TOKEN]
        max_len = min(max(seq_lens)+1, self.max_len)
        input_dict_batch = []
        for tokens in seqs:
            tokens = [START_TOKEN] + tokens
            tokens = tokens[:max_len]
            input_ids, offsets = self.mismatched_tokenizer.encode(tokens)
            input_dict = self.build_input_dict(input_ids, offsets, len(tokens))
            input_dict_batch.append(input_dict)
        batch_input_dict = self.collate_fn(input_dict_batch)
        for k, v in batch_input_dict.items():
            batch_input_dict[k] = v.to(self.device)
        return batch_input_dict
    
    def postprocess(self, batch, batch_label_probs, batch_label_ids, batch_incor_probs):
        keep_id = self.vocab.correct_vocab["tag2id"][KEEP_LABEL]
        all_results = []
        for tokens, label_probs, label_ids, incor_prob in zip(batch, batch_label_probs, 
                                                            batch_label_ids, batch_incor_probs):
            # since we add special tokens before truncation, max_len should minus 1. This is different from original gector.
            length = min(len(tokens), self.max_len - 1)
            edits = []

            # skip the whole sent if all labels are $KEEP
            if max(label_ids) == keep_id:
                all_results.append(tokens)
                continue

            # if max detect_incor_probs < min_error_prob, skip
            if incor_prob < self.min_error_probability:
                all_results.append(tokens)
                continue
            
            for idx in range(length + 1):
                if idx == 0:
                    token = START_TOKEN
                else:
                    # tokens in ori_batch don't have "$START" token, thus offset = 1
                    token = tokens[idx-1]
                if label_ids[idx] == keep_id:
                    continue
                if re.search("\s+", token): # prediction for \s matched token is $keep, for spellcheck task.
                    continue
                label = self.vocab.correct_vocab["id2tag"][label_ids[idx]]
                action = self.get_label_action(token, idx, label_probs[idx], label)
                
                if not action:
                    continue
                edits.append(action)
            # append the target sent (list of target tokens)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                        prev_preds_dict):
        
        new_pred_ids = []
        total_updated = 0
        
        for i, ori_id in enumerate(pred_ids):
            ori_tokens = final_batch[ori_id]
            pred_tokens = pred_batch[i]
            prev_preds = prev_preds_dict[ori_id]

            if ori_tokens != pred_tokens:
                if pred_tokens not in prev_preds:
                    final_batch[ori_id] = pred_tokens
                    new_pred_ids.append(ori_id)
                    prev_preds_dict[ori_id].append(pred_tokens)
                else:
                    final_batch[ori_id] = pred_tokens
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated


    def get_label_action(self, token: str, idx: int, label_prob: float, label: str):
        if label_prob < self.min_error_probability or label in [UNK_LABEL, PAD_LABEL, KEEP_LABEL]:
            return None

        if label.startswith("$REPLACE_") or label.startswith("$TRANSFORM_") or label == "$DELETE":
            start_pos = idx
            end_pos = idx + 1
        elif label.startswith("$APPEND_") or label.startswith("$MERGE_"):
            start_pos = idx + 1
            end_pos = idx + 1
        
        if label == "$DELETE":
            processed_label = ""
        
        elif label.startswith("$TRANSFORM_") or label.startswith("$MERGE_"):
            processed_label = label[:]
        
        else:
            processed_label = label[label.index("_")+1:]
        return start_pos - 1, end_pos - 1, processed_label, label_prob

    def build_input_dict(self, input_ids, offsets, word_level_len):
        token_type_ids = [0 for _ in range(len(input_ids))]
        attn_mask = [1 for _ in range(len(input_ids))]
        word_mask = [1 for _ in range(word_level_len)]

        input_dict = {
            "input_ids": input_ids, 
            "token_type_ids": token_type_ids, 
            "attention_mask": attn_mask, 
            "offsets": offsets,
            "word_mask": word_mask}
        return input_dict

    def fix_seed(self):
        torch.manual_seed(1)
        if not self.use_amp:
            torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed(43)
