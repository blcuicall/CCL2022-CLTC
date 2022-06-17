import torch
from transformers import BertTokenizer
from utils import *
from model import BERT_Model
from tqdm import tqdm
import os
import argparse


class Decoder:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.test_loader = init_dataloader(config.test_path, config, "test", self.tokenizer)
        self.model = BERT_Model(config, self.test_loader.dataset)
        self.model.to(self.device)
        self.config = config

    def __forward_prop(self, dataloader, back_prop=True):
        collected_outputs = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, logits = self.model(**batch)
            outputs = torch.argmax(logits, dim=-1)
            for outputs_i in outputs:
                collected_outputs.append(outputs_i)
        return collected_outputs

    def decode(self):
        model = self.model
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        with torch.no_grad():
            outputs = self.__forward_prop(dataloader=self.test_loader, back_prop=False)
            save_decode_result_lbl(outputs, self.test_loader.dataset.data, self.config.save_path)
            # save_decode_result_para(outputs, self.test_loader.dataset.data, self.config.save_path)


def main(config):
    decoder = Decoder(config)
    decoder.decode()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)

    args = parser.parse_args()
    main(args)
