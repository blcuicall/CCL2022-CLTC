# -*- coding:UTF-8 -*-
"""
A tweak version of Allennlp's pretrained_transformer_mismatched_indexer/embedder
"""
import torch

class MisMatchedTokenizer:
    def __init__(self, tokenizer, max_len, max_pieces_per_token=None, special_start_token_ids=[], special_end_token_ids=[]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_pieces_per_token = max_pieces_per_token
        self.special_start_token_ids = special_start_token_ids
        self.special_end_token_ids = special_end_token_ids

    def encode(self, words: list, add_special_tokens=False):
        input_ids = []
        offsets = []
        for word in words:
            wordpieces = self.tokenizer.tokenize(word)
            wordpiece_ids = [self.tokenizer.vocab[wordpiece] for wordpiece in wordpieces]
            # we set ovv token's wordpiece id to unk_token_id, 
            # thus we can deal with it as normal tokens
            if not len(wordpiece_ids):
                wordpiece_ids = [self.tokenizer.unk_token_id]
            elif (self.max_pieces_per_token is not None):
                wordpiece_ids = wordpiece_ids[:self.max_pieces_per_token]
            offsets.append((len(input_ids), len(input_ids)+len(wordpiece_ids)-1))
            input_ids.extend(wordpiece_ids)
        if add_special_tokens:
            offsets = self._increment_offsets(offsets, len(self.special_start_token_ids))
            input_ids = self._add_special_tokens(input_ids)
        return input_ids, offsets

    def _add_special_tokens(self, input_ids):
        return self.special_start_token_ids + input_ids + self.special_end_token_ids

    def _increment_offsets(self, offsets, increment):
        return [(offset[0]+ increment, offset[1]+increment) for offset in offsets]

class MisMatchedSampleIndexer:
    def __init__(self, input_pad_id):
        self.input_pad_id = input_pad_id

    def build_input_dict(self, input_ids, offsets, word_level_len):
        token_type_ids = [0 for _ in range(len(input_ids))]
        attn_mask = [1 for _ in range(len(input_ids))]
        word_mask = [1 for _ in range(word_level_len)]
        # correspond to IndexedTokenList
        input_dict = {
            "input_ids": input_ids, 
            "token_type_ids": token_type_ids, 
            "attention_mask": attn_mask, 
            "offsets": offsets,
            "word_mask": word_mask}
        return input_dict

class MisMatchedEmbedder:
    def __init__(self, device, sub_token_mode):
        self.device = device
        self.sub_token_mode = sub_token_mode

    # build span embeddings
    def _batched_index_select(self, wordpiece_embeddings, span_indices):
        """
        this function returns selected values in the target with respect to the provided indices
        Params:
        wordpiece_embeddings: (bsz, subwordlevel_seq_length, hidden_size)
        span_idxs: (bsz, wordlevel_seq_length, hidden_size)
        
        """
        flattened_indices = self._flatten_and_batch_shift_indices(span_indices, wordpiece_embeddings.size(1))
        flattened_selected_embeddings = wordpiece_embeddings.view(-1, wordpiece_embeddings.size(-1)).index_select(0, flattened_indices)

        # get the desired shape: (bsz, seq_len, span_width, hidden_size)
        selected_shape = list(span_indices.size()) + [wordpiece_embeddings.size(-1)]
        selected_embeddings = flattened_selected_embeddings.view(*selected_shape)
        return selected_embeddings

    def _flatten_and_batch_shift_indices(self, span_indices, seq_len):
        # [0,bsz*seq_len], shape: (bsz,)
        offsets = torch.arange(span_indices.size(0), dtype=torch.long).to(self.device) * seq_len
        # shape: (bsz, 1, 1)
        # this operation maps the dim of span_indices
        for _ in range(len(span_indices.size())-1):
            offsets = offsets.unsqueeze(1)

        # indices are shifted considering the idx in the whole batch, i.e,
        # for the first word in the second sequence, its idx is shifted by 
        # idx + seq_length
        # shape: (bsz, seq_len, span_width)
        offset_indices = span_indices + offsets
        # (bsz * seq_len)
        offset_indices = offset_indices.view(-1)
        # note that for offset padding idxs, we still need masks as they're wrong for indexing.
        return offset_indices

    def _batched_span_select(self, wordpiece_embeddings, offset_spans):
        span_embeddings, span_mask = [], []
        """
        Returns: 
        span_embeddings : `torch.Tensor`
            A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
            representing the embedded spans extracted from the batch flattened target tensor.
        span_mask: `torch.BoolTensor`
            A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
            the returned span embeddings.
        """
        # (batch_size, num_spans, 1)
        span_starts, span_ends = offset_spans.split(1, dim=-1)
        # actually, the widths should be width + 1 since for single-wordpiece word, span_end - span_start = 0
        # but we only need the max one, so there's no need to add 1 for every width in span_widths.
        # we operate it in the next line of code.
        span_widths = span_ends - span_starts 
        # Thus, here we +1 to get the actual max_batch_span_width
        # span_widths.max().item() + 1
        # (1, 1, max_batch_span_width)
        max_span_range_indices = torch.arange(0, span_widths.max().item() + 1, dtype=torch.long).to(self.device).view(1, 1, -1)
        
        # we create a range vector of size max_span_width, and mask walues 
        # which are greater than the actual length of the span
        span_mask = max_span_range_indices <= span_widths
        raw_span_indices = span_starts + max_span_range_indices
        # for some spans which are near the end of the sequence, 
        # they may have a start_idx + max_batch_span_width > wordpiece_length
        # this is different from allennlp's implementation, which also takes (-1,-1) offsets into account
        # where we replace special tokens to [UNK], which becomes normal offsets in later procedure.
        span_mask = span_mask & (raw_span_indices < wordpiece_embeddings.size(1))
        # mask invalid values which are mentioned above
        span_indices = raw_span_indices * span_mask
        span_embeddings = self._batched_index_select(wordpiece_embeddings, span_indices)
        return span_embeddings, span_mask

    def _get_padding_mask(self, span_mask, word_mask):
        """
        Params: 
        span_mask: (bsz, seq_len, span_width)
        word_mask: (bsz, seq_len)
        
        Returns:
        padding_mask: (bsz, seq_len, span_width, 1)
        """
        padding_mask = word_mask.view(*word_mask.size(), 1).bool() & span_mask
        padding_mask.unsqueeze_(-1)
        return padding_mask

    def get_span_embeddings(self, wordpiece_embeddings, offsets, word_mask):
        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = self._batched_span_select(wordpiece_embeddings.contiguous(), offsets)

        #  span_masks only deal with span padding，but we also need seq padding which can be done with word mask
        padding_mask = self._get_padding_mask(span_mask, word_mask)
        # zero out paddings
        span_embeddings *= padding_mask
        return span_embeddings, padding_mask

    def get_mismatched_embeddings(self, wordpiece_embeddings, offsets, word_mask):
        span_embeddings, padding_mask = self.get_span_embeddings(wordpiece_embeddings, offsets, word_mask)
        if self.sub_token_mode == "first":
            mismatched_embeddings = span_embeddings[:, :, 0, :]
        elif self.sub_token_mode == "average":
            mismatched_embeddings = span_embeddings.sum(2) / torch.clamp(padding_mask.sum(2), min=1)
        else:
            raise NotImplementedError()
        return mismatched_embeddings
