# -*- coding: utf-8 -*-
# Author: Cunliang Kong
# Affiliation: BLCU-ICALL

from typing import Dict, List, NamedTuple, Optional

import torch
from torch import Tensor
from fairseq.models import FairseqEncoder

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)


class HuggingFaceBartEncoder(FairseqEncoder):

    def __init__(self, args, task, encoder):
        super().__init__(task.target_dictionary)
        self.encoder = encoder
        self.pad_idx = task.source_dictionary.pad()
        self.bos_idx = task.source_dictionary.bos()
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        attention_mask = src_tokens.ne(self.pad_idx).float()
        encoder_outputs = self.encoder(
            input_ids=src_tokens,
            attention_mask=attention_mask,
        )

        src_lengths = (src_tokens.ne(self.pad_idx).sum(
            dim=1, dtype=torch.int32).reshape(-1, 1).contiguous())

        return EncoderOut(
            encoder_out=[encoder_outputs[0]],
            encoder_padding_mask=[attention_mask],
            encoder_embedding=[],
            encoder_states=[],
            src_tokens=[src_tokens],
            src_lengths=[src_lengths],
        )

    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]],
                            new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out.encoder_out) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [
                encoder_out.encoder_out[0].index_select(0, new_order)
            ]
        if len(encoder_out.encoder_padding_mask) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out.encoder_padding_mask[0].index_select(0, new_order)
            ]
        if len(encoder_out.encoder_embedding) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out.encoder_embedding[0].index_select(0, new_order)
            ]

        if len(encoder_out.src_tokens) == 0:
            src_tokens = []
        else:
            src_tokens = [
                (encoder_out.src_tokens[0]).index_select(0, new_order)
            ]

        if len(encoder_out.src_lengths) == 0:
            src_lengths = []
        else:
            src_lengths = [
                (encoder_out.src_lengths[0]).index_select(0, new_order)
            ]

        encoder_states = encoder_out.encoder_states
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(0, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        return self.args.max_source_positions
