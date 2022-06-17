# -*- coding:UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from utils.mismatched_utils import MisMatchedEmbedder
from transformers import AutoModel
import torch
import os

class SeqEncoder(nn.Module):
    def __init__(self, sub_token_mode, encoder_path, device):
        super().__init__()
        self.matched_embedder = AutoModel.from_pretrained(encoder_path)
        self.hidden_size = self.matched_embedder.config.hidden_size
        self.mismatched_embedder = MisMatchedEmbedder(device, sub_token_mode)
        self.activate_grad = True
    
    def forward(self, input_dict, requires_grad=True):
        if self.activate_grad != requires_grad:
            for param in self.parameters():
                param.requires_grad_(requires_grad)
            self.activate_grad = requires_grad

        output_dict = self.matched_embedder(
            input_ids=input_dict["input_ids"],
            token_type_ids=input_dict["token_type_ids"],
            attention_mask=input_dict["attention_mask"],
        )
        last_hidden_states = output_dict[0]

        word_embeddings = self.mismatched_embedder.get_mismatched_embeddings(
            last_hidden_states,
            offsets=input_dict["offsets"],
            word_mask=input_dict["word_mask"])
        return word_embeddings

class GECToRModel(nn.Module):
    def __init__(self, 
        encoder_path, 
        num_detect_tags, 
        num_correct_tags, 
        additional_confidence,
        dp_rate, 
        detect_pad_id, 
        correct_pad_id,
        detect_incorrect_id, 
        correct_keep_id,
        sub_token_mode, 
        device
        ):
        
        super().__init__()
        self.device = device
        self.detect_incorrect_id = detect_incorrect_id
        self.correct_keep_id = correct_keep_id
        self.num_correct_tags = num_correct_tags
        self.num_detect_tags = num_detect_tags
        self.additional_confidence = additional_confidence
        
        self.encoder = SeqEncoder(sub_token_mode, encoder_path, device)
        self.embedding_size = self.encoder.hidden_size
        self.detect_proj_layer = nn.Linear(self.embedding_size, num_detect_tags)
        self.correct_proj_layer = nn.Linear(self.embedding_size, num_correct_tags)
        self.dropout = nn.Dropout(dp_rate, inplace=True)
        self.detect_loss_fn = CrossEntropyLoss(ignore_index=detect_pad_id)
        self.correct_loss_fn = CrossEntropyLoss(ignore_index=correct_pad_id)
        
    def forward(self, input_dict, encoder_requires_grad=True):
        embeddings = self.encoder(input_dict, encoder_requires_grad)
        batch_size, seq_len = embeddings.shape[:-1]

        correct_logits = self.correct_proj_layer(self.dropout(embeddings))
        detect_logits = self.detect_proj_layer(embeddings)

        correct_probs = F.softmax(correct_logits, dim=-1)
        detect_probs = F.softmax(detect_logits, dim=-1)
        # shape: (bsz, seq_len)
        detect_incorrect_probs = detect_probs[:, :, self.detect_incorrect_id] * input_dict["word_mask"] 
        # shape: (bsz, ), 句子中incorrect标签的最大的概率
        max_incorrect_probs = torch.max(detect_incorrect_probs, dim=-1).values
        if self.additional_confidence != 0:
            correct_probs_change = torch.zeros(batch_size, seq_len, self.num_correct_tags, dtype=torch.float32).to(self.device)
            correct_probs_change[:, :, self.correct_keep_id] = self.additional_confidence
            correct_probs += correct_probs_change

        total_loss = None
        if "detect_tag_ids" in input_dict and "correct_tag_ids" in input_dict:
            correct_tag_target_ids = input_dict["correct_tag_ids"]
            detect_tag_target_ids = input_dict["detect_tag_ids"]
            correct_loss = self.correct_loss_fn(correct_logits.view(-1, self.num_correct_tags), correct_tag_target_ids.view(-1))
            detect_loss = self.detect_loss_fn(detect_logits.view(-1, self.num_detect_tags), detect_tag_target_ids.view(-1))
            total_loss = correct_loss + detect_loss
        output_dict = {"logits_labels": correct_logits,
                    "logits_d_tags": detect_logits,
                    "class_probabilities_labels": correct_probs,
                    "class_probabilities_d_tags": detect_probs,
                    "max_error_probability": max_incorrect_probs}
        output_dict["loss"] = total_loss

        return output_dict
