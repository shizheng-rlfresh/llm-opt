import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from typing import List, Optional, Type
class customDataLoader:
    def __init__(self, 
                 dataset = Dataset,
                 tokenizer = AutoTokenizer,
                 randomSeed: Optional[int] = None, 
                 batchSize: Optional[int] = None, 
                 dynamicPadding: Optional[bool] = None,
                 shuffle: Optional[bool] = None) -> None:
        if not hasattr(dataset, 'shape') or not hasattr(dataset, 'features'):
            raise AttributeError("The dataset provided does not have the required 'shape' or 'features' attributes.")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.randomSeed = randomSeed
        self.batchSize = batchSize
        self.dynamicPadding = dynamicPadding
        self.shuffle = shuffle
        # record generation stream
        self.idxStream = [i for i in range(self.dataset.shape[0])]
        self.idxPointer = 0
        # dataset features, e.g., input_ids, attention_masks, labels
        self.features = list(self.dataset.features)

        # initialization
        if not self.randomSeed and self.randomSeed != 0:
            self.randomSeed = 42 # for reproducibility
        torch.manual_seed(self.randomSeed)
        np.random.seed(self.randomSeed)
        # random shuffle
        if self.shuffle:
            np.random.shuffle(self.idxStream)

        if not self.dynamicPadding:
            self.dynamicPadding = True
        if not self.batchSize:
            self.batchSize = 64 # depends on memory budget and hyper-param choice
    
    def padding(self, batch: dict, 
                max_length: Optional[int] = None,
                paddingSize: Optional[str] = 'left') -> dict:

        padding_token_id = self.tokenizer.pad_token_id

        # dynamic padding uses max length in the minibatch
        if self.dynamicPadding:
            max_length = max([len(sbatch) for sbatch in batch["input_ids"]])
        if paddingSize == "right":
            for k, v in batch.items():
                # for attention_masks, set padding token to be `0`
                if k == "attention_mask":
                    batch[k] = torch.tensor([vi + [0] * (max_length - len(vi)) for vi in v])
                # for input_ids, labels, or other features
                else:
                    batch[k] = torch.tensor([vi + [padding_token_id] * (max_length - len(vi)) for vi in v])
        # for autoregressive, i.e., decoder-only, models, use left padding
        else:
            for k, v in batch.items():
                # for attention_masks, set padding token to be `0`
                if k == "attention_mask":
                    batch[k] = torch.tensor([[0] * (max_length - len(vi)) + vi for vi in v])
                 # for input_ids, labels, or other features
                else:
                    batch[k] = torch.tensor([[padding_token_id] * (max_length - len(vi)) + vi for vi in v])
        return batch
    
    def generate(self) -> dict:
        batch_idx = self.idxStream[self.idxPointer * self.batchSize:(self.idxPointer + 1) * self.batchSize]
        batch = self.padding(self.dataset[batch_idx])

        # a new epoch
        if (self.idxPointer + 1) * self.batchSize > len(self.idxStream):
            self.idxPointer = 0
        else:
            self.idxPointer += 1
            
        return batch 