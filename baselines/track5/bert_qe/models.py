import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder

from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder
from torch.autograd import Variable
import numpy as np




class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.pred_model = bert_model
        self.proj_hidden = nn.Linear(self.bert_hidden_dim, 1)


    def forward(self, inp_tensor, msk_tensor, seg_tensor):
        inputs_hidden, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        score = F.sigmoid(self.proj_hidden(inputs))
        return score





















