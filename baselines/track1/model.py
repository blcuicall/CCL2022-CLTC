import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BERT_Model(nn.Module):

    def __init__(self, config, freeze_bert=False, tie_cls_weight=False):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.label_ignore_id = config.label_ignore_id

        if tie_cls_weight:
            self.tie_cls_weight()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            trg_ids=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = self.classifier(bert_output.last_hidden_state)
        loss = None
        if trg_ids is not None:
            loss_function = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
            loss = loss_function(logits.view(-1, self.bert.config.vocab_size), trg_ids.view(-1))
        return loss, logits
    
    def tie_cls_weight(self):
            self.classifier.weight = self.bert.embeddings.word_embeddings.weight













