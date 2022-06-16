# -*- coding: utf-8 -*-
# Author: Cunliang Kong
# Affiliation: BLCU-ICALL

from typing import Dict, List, Optional

import torch
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor


class HuggingFaceBartDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, task, decoder, lm_head):
        super().__init__(task.target_dictionary)
        self.decoder = decoder
        self.args = args
        self.task = task
        self.lm_head = lm_head
        self.pad_idx = task.target_dictionary.pad()

    def forward(
        self,
        prev_output_tokens,
        src_lengths=None,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        features = self.extract_features(prev_output_tokens, incremental_state,
                                         encoder_out)
        lm_logits = self.lm_head(features)
        return (lm_logits, )

    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
    ):
        if incremental_state:
            past_key_values = self.get_incremental_state("past_key_values")
            use_cache = True
        else:
            past_key_values = None
            use_cache = False

        attention_mask = prev_output_tokens.ne(self.pad_idx).float()

        outputs = self.decoder(
            input_ids=prev_output_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_out.encoder_out[0],
            encoder_attention_mask=encoder_out.encoder_padding_mask[0],
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        last_hidden_states = outputs[0]

        if incremental_state:
            self.set_incremental_state(incremental_state, "past_key_values",
                                       outputs[1])
        return last_hidden_states

    def max_positions(self):
        return self.args.max_target_positions

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        pass
