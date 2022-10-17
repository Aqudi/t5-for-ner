import argparse

from utils.set_seed import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="")
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
parser.add_argument("--model_name_or_path", default="google/mt5-base")
parser.add_argument("--tokenizer_name_or_path", default="")
parser.add_argument("--cached_dataset_path", default="cached_dataset")
parser.add_argument("--max_input_length", default=512, type=int, required=False)
parser.add_argument("--max_target_length", default=128, type=int, required=False)
parser.add_argument("--learning_rate", default=3e-4, type=int)
parser.add_argument("--weight_decay", default=0.0, type=int)
parser.add_argument("--adam_epsilon", default=1e-8, type=int)
parser.add_argument("--warmup_steps", default=0, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--max_epochs", default=15, type=int)
parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
parser.add_argument("--max_grad_norm", default=1.0, type=int)
parser.add_argument("--seed", default=928, type=int)
parser.add_argument("--accelerator", default="gpu")
parser.add_argument("--devices", default=1, type=int)
parser.add_argument("--debug", default=False, action="store_true")


args = parser.parse_args()

args_dict = vars(args)
if args_dict.get("tokenizer_name_or_path") == "":
    args_dict.update(dict(tokenizer_name_or_path=args.model_name_or_path))

set_seed(args_dict.get("seed"))
