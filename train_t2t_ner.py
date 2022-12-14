import os

import datasets
from setproctitle import setproctitle
from transformers import (AutoModelForSeq2SeqLM, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5TokenizerFast)
from transformers.data.data_collator import DataCollatorForSeq2Seq

from data import load_t2t_cross_validation_dataset, shuffle_and_save_file
from datamodule import T5NerFineTunerDataModule

def train(dataset, args, cross_epoch=None):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    tokenizer = T5TokenizerFast.from_pretrained(args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    logging_steps = len(dataset["train"]) // args.batch_size

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
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
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

    from utils.set_seed import set_seed

    setproctitle("train-t5-ner")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(928)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        default="./data_learn/preprocessed/klue_ner_t2t_train.tsv",
        required=False,
    )
    parser.add_argument(
        "--test_dataset",
        default="./data_learn/preprocessed/klue_ner_t2t_test.tsv",
        required=False,
    )
    parser.add_argument("--cross-validation", required=False, action="store_true")
    parser.add_argument("--fold", required=False, default=5)
    parser.add_argument("--checkpoint", required=False, default="google/mt5-base")
    parser.add_argument("--output_dir", required=False, default="google/mt5-base")
    parser.add_argument("--batch_size", type=int, required=False, default=12)
    parser.add_argument("--num_train_epochs", type=int, required=False, default=10)
    parser.add_argument("--learning_rate", type=float, required=False, default=2e-5)
    parser.add_argument("--max_input_length", default=512, type=int, required=False)
    parser.add_argument("--max_target_length", default=128, type=int, required=False)
    parser.add_argument("--early_stopping_patience", default=2, required=False)

    args = parser.parse_args()

    print(args.__dict__)

    if args.cross_validation:
        shuffled_data_path = "shuffled_train_data.txt"
        shuffle_and_save_file(args.train_dataset, shuffled_data_path)

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
        datamodule = T5NerFineTunerDataModule(
            tokenizer_name_or_path=args.checkpoint,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            batch_size=args.batch_size,
            train_dataset=args.train_dataset,
            test_dataset=args.test_dataset,
        )
        datamodule.prepare_data()
        datamodule.setup()
        dataset = datasets.DatasetDict(
            {
                "train": datamodule.train_dataset,
                "val": datamodule.val_dataset,
            },
        )
        print("-------- val_split_dataset--------\n", dataset)

        train(dataset=dataset, args=args)
