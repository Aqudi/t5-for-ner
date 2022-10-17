import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from data import load_t2t_dataset

from transformers import (
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk

import pytorch_lightning as pl


class T5NerFineTunerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name_or_path,
        max_seq_length,
        batch_size,
        cached_dataset_path,
        **kwargs,
    ):
        super(T5NerFineTunerDataModule, self).__init__()
        self.save_hyperparameters()

        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.hparams.tokenizer_name_or_path
        )
        self.cached_dataset_path = cached_dataset_path

    def setup(self, stage=None):
        dataset = load_from_disk(self.cached_dataset_path)
        train_val = dataset.get("train").train_test_split(test_size=0.2)
        self.train_dataset = train_val.get("train")
        self.val_dataset = train_val.get("test")
        self.test_dataset = dataset.get("test")

        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")
        self.test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["sentence"],
            max_length=self.hparams.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        labels = self.tokenizer(
            examples["label"],
            max_length=self.hparams.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_data(self):
        self.raw_dataset = load_t2t_dataset()
        if os.path.exists(self.cached_dataset_path):
            print("캐시된 데이터셋 사용", flush=True)
            return

        dataset = self.raw_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["sentence", "label"],
        )

        dataset.set_format(type="torch")
        dataset.save_to_disk(self.cached_dataset_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
        )
