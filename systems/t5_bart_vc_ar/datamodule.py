import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments

import Define
from .collate import CodecCollate


class DataModule(object):
    def __init__(self, data_config: dict, model_config, train_config: Seq2SeqTrainingArguments):
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config

        self.collate = CodecCollate(data_config, model_config, train_config)
    
    def train_dataset(self) -> Dataset:
        train_dataset = load_dataset(
            self.data_config["name"],
            split="+".join(self.data_config["train_splits"]),
            cache_dir=Define.HF_CACHE_DIR
        )

        if Define.DEBUG:
            train_dataset = train_dataset.select(list(range(1000)))
        train_dataset = train_dataset.map(
            self.collate.map_fn,
            remove_columns=train_dataset.column_names,
            batched=True,
            batch_size=self.train_config.per_device_train_batch_size,
        )

        return train_dataset

    def eval_dataset(self) -> Dataset:
        eval_dataset = load_dataset(
            self.data_config["name"],
            split="+".join(self.data_config["eval_splits"]),
            cache_dir=Define.HF_CACHE_DIR
        )
        # Inference is slow
        eval_dataset = eval_dataset.train_test_split(test_size=400)["test"]

        if Define.DEBUG:
            eval_dataset = eval_dataset.select(list(range(10)))
        eval_dataset = eval_dataset.map(
            self.collate.map_fn,
            remove_columns=eval_dataset.column_names,
            batched=True,
            batch_size=self.train_config.per_device_eval_batch_size,
        )

        return eval_dataset
    
    def data_collator(self):
        return self.collate.collate_fn
