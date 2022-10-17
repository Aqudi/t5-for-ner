import os, sys

sys.path.append("/home/work/team01")

from data import (
    load_t2t_dataset,
    load_t2t_cross_validation_dataset,
    shuffle_and_save_file,
    t2t_train_dataset_path,
)
import datasets
from transformers import (
    T5TokenizerFast,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    EarlyStoppingCallback,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import Trainer, Seq2SeqTrainer
import numpy as np
import pandas as pd
import evaluate
import torch
from seqeval import metrics as seqeval_metrics
from torch import optim
from setproctitle import setproctitle


def get_tokenized_dataset(dataset, tokenizer, max_input_length, max_target_length):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["sentence"], max_length=max_input_length, truncation=True
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["label"], max_length=max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    print("--------tokenized_datasets--------\n", tokenized_datasets)
    # remove the columns with strings
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "label"])
    return tokenized_datasets


def train(dataset, args, cross_epoch=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    tokenizer = T5TokenizerFast.from_pretrained(args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    tokenized_datasets = get_tokenized_dataset(
        dataset,
        tokenizer,
        args.max_input_length,
        args.max_target_length,
    )

    logging_steps = len(tokenized_datasets["train"]) // args.batch_size

    model_name = f"checkpoints/{args.output_dir}"
    if args.cross_validation and cross_epoch != None:
        model_name += f"-cross-{cross_epoch}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=500,
        weight_decay=0.01,
        save_total_limit=3,
        predict_with_generate=True,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
        # optimizers= (optim.Adam, None)
    )

    print("#### train ######")
    trainer.train()

    print("#### evaluate ######")
    trainer.evaluate()


if __name__ == "__main__":
    import argparse
    import json

    from utils.set_seed import set_seed

    setproctitle("0ys_ulm-summary")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(928)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-validation", required=False, default=False, action="store_true"
    )
    parser.add_argument("--fold", required=False, default=5)
    parser.add_argument("--checkpoint", required=False, default="./model/kt-ulm-small")
    parser.add_argument("--output_dir", required=False, default="kt-ulm-small")
    parser.add_argument("--batch_size", type=int, required=False, default=12)
    parser.add_argument("--num_train_epochs", type=int, required=False, default=10)
    parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    parser.add_argument("--max_input_length", default=512, type=int, required=False)
    parser.add_argument("--max_target_length", default=128, type=int, required=False)
    parser.add_argument("--early_stopping_patience", default=2, required=False)
    parser.add_argument("--cross_validatione", default=False, required=False)

    args = parser.parse_args()

    max_input_length = args.max_input_length
    max_target_length = args.max_target_length

    print(args.__dict__)

    if args.cross_validation:
        shuffled_data_path = "shuffled_train_data.txt"
        shuffle_and_save_file(t2t_train_dataset_path, shuffled_data_path)

        dataset = load_t2t_cross_validation_dataset(
            path=shuffled_data_path, fold=args.fold
        )
        print("-------- val_split_dataset--------\n", dataset)

        for i in range(args.fold):
            print("_______________CrossVal Epoch: ", i)
            setproctitle(f"cross-epoch{i}")

            cross_dataset = datasets.DatasetDict(
                {"train": dataset["train"][i], "val": dataset["val"][i]}
            )
            print("-------- cross_dataset--------\n", cross_dataset)
            train(dataset=cross_dataset, args=args, cross_epoch=i)
    else:
        train_data = load_t2t_dataset()["train"]
        train_val_data = train_data.train_test_split(test_size=0.2)
        dataset = datasets.DatasetDict(
            {
                "train": train_val_data["train"],
                "val": train_val_data["test"],
            },
        )
        print("-------- val_split_dataset--------\n", dataset)

        train(dataset=dataset, args=args)
