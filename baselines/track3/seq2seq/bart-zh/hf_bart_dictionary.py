# -*- coding: utf-8 -*-
# Author: Cunliang Kong
# Affiliation: BLCU-ICALL
import torch
from fairseq import utils
from fairseq.data import Dictionary


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re

        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(" +", " ", sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == " ##":
        sentence = sentence.replace(symbol, "").strip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


class HuggingFaceBartDictionary(Dictionary):

    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = len(self.symbols)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)

        d.bos_word = "[CLS]"
        d.unk_word = "[UNK]"
        d.pad_word = "[PAD]"
        d.eos_word = "[SEP]"
        d.bos_index = d.add_symbol("[CLS]")
        d.unk_index = d.add_symbol("[UNK]")
        d.pad_index = d.add_symbol("[PAD]")
        d.eos_index = d.add_symbol("[SEP]")
        return d

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
        separator=" ",
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(
                    t,
                    bpe_symbol,
                    escape_unk,
                    extra_symbols_to_ignore,
                    include_eos=include_eos,
                ) for t in tensor)

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        if not include_eos:
            extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = separator.join(
            token_string(i) for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore)

        return post_process(sent, bpe_symbol)
