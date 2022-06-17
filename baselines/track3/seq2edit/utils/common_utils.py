from contextlib import contextmanager
import torch
from utils.mismatched_utils import *
from src.dataset import MyCollate, Seq2EditDataset
from torch.utils.data import DataLoader
import json

def read_config(path):
    with open(path, "r", encoding="utf8") as fr:
        config = json.load(fr)
    return config       

def init_dataloader(subset,
                    data_path, 
                    use_cache,
                    tokenizer, 
                    vocab, 
                    input_pad_id,
                    detect_pad_id,
                    correct_pad_id,
                    max_len, 
                    batch_size,
                    tag_strategy, 
                    skip_complex, 
                    skip_correct=0, 
                    tp_prob=1, 
                    tn_prob=1):

    my_collate_fn = MyCollate(input_pad_id, detect_pad_id, correct_pad_id)

    sub_dataset = Seq2EditDataset(data_path, 
                                use_cache,
                                tokenizer, 
                                vocab, 
                                max_len, 
                                tag_strategy,
                                skip_complex,
                                skip_correct,
                                tp_prob,
                                tn_prob)

    if subset == "train":
        shuffle = True
    else:
        shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(sub_dataset, shuffle=shuffle, drop_last=True)
    data_loader = DataLoader(
        sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=my_collate_fn,
        num_workers=0,
        sampler=sampler
        )
    return data_loader

@contextmanager
def torch_distributed_master_process_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

