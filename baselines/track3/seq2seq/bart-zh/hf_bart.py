# -*- coding: utf-8 -*-
# Author: Cunliang Kong
# Affiliation: BLCU-ICALL

import logging

from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from transformers import BartForConditionalGeneration
from .hf_bart_encoder import HuggingFaceBartEncoder
from .hf_bart_decoder import HuggingFaceBartDecoder

logger = logging.getLogger(__name__)


@register_model("hf_bart")
class HuggingFaceBartModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--hf-model-name',
                            type=str,
                            metavar='S',
                            help='huggingface model name')

    @classmethod
    def build_model(cls, args, task):
        hf_bart_model = BartForConditionalGeneration.from_pretrained(
            args.hf_model_name)
        task.config = hf_bart_model.config
        # hf_bart_model.resize_token_embeddings(len(task.src_dict))
        encoder = HuggingFaceBartEncoder(args, task,
                                         hf_bart_model.get_encoder())
        decoder = HuggingFaceBartDecoder(args, task,
                                         hf_bart_model.get_decoder(),
                                         hf_bart_model.lm_head)
        return cls(encoder, decoder)


@register_model_architecture("hf_bart", "hf_bart")
def hf_t5_architecture(args):
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
