import os

from datasets import load_dataset, DatasetDict

from data.preprocess import t2t_test, t2t_train


def load_t2t_dataset():
    dataset = load_dataset(
        "csv",
        column_names=["sentence", "label"],
        delimiter="\t",
        data_files={
            "train": t2t_train,
            "test": t2t_test,
        },
    )
    return dataset


def shuffle_and_save_file(path, destination_path):
    if os.path.exists(destination_path):
        print("셔플된 파일이 있습니다.")
        return

    import random

    lines = None
    with open(path, encoding="utf8") as f:
        print("파일 읽어옴")
        lines = f.readlines()
        print(lines[:3])

    random.shuffle(lines)
    with open(destination_path, encoding="utf8", mode="wt") as f:
        print("파일 섞어서 저장")
        f.writelines(lines)
        print(lines[:3])


def load_t2t_cross_validation_dataset(path=None, fold=5):
    if not path:
        path = t2t_train

    percent = 100 // fold
    train_data = load_dataset(
        "csv",
        column_names=["sentence", "label"],
        delimiter="\t",
        data_files={
            "train": path,
        },
        split=[f"train[:{k}%]+train[{k+percent}%:]" for k in range(0, 100, percent)],
    )
    val_data = load_dataset(
        "csv",
        column_names=["sentence", "label"],
        delimiter="\t",
        data_files={
            "train": path,
        },
        split=[f"train[{k}%:{k+percent}%]" for k in range(0, 100, percent)],
    )
    ds = DatasetDict({"train": train_data, "val": val_data})
    return ds
