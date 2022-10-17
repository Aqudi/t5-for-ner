import re
import os


tags = ["QT", "PS", "LC", "DT", "TI", "OG"]
tag_descriptions = ["수량", "사람", "장소", "날짜", "시간", "기관"]


def readlines(fileapth):
    exmaples = None
    with open(fileapth, encoding="utf8") as f:
        exmaples = f.readlines()
    return exmaples


def save(filepath, lines):
    directory = "/".join(filepath.split("/")[:-1])
    if not os.path.exists(directory):
        print(directory, "is not exsits")
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "wt", encoding="utf8") as f:
        for line in lines[:-1]:
            f.write(line)
            f.write("\n")
        f.write(lines[-1])


def create_examples_by_tags(original_example, labels):
    results = []
    label_by_tags = {tag: [] for tag in tags}
    for label in labels:
        label_by_tags[label[1]].append(f"{label[1]}:{label[0]}")

    for tag, desc in zip(tags, tag_descriptions):
        current_labels = label_by_tags.get(tag)
        label_str = "O"
        if current_labels:
            label_str = ",".join(current_labels)
        example = f"[sentence] {original_example.strip()} [tag] {tag}:{desc}\t[label] {label_str}"
        results.append(example)
    return results


def preprocess_data_for_t2t(filepath, destination_filepath):
    target = re.compile("<(.*?):([A-Z]*?)>")

    results = []

    examples = readlines(filepath)
    for example in examples:
        example = str(example)
        # 패턴 내의 그룹 조회 /index
        example_only = re.sub(target, r"\1", example)
        labels = re.findall(target, example)

        results.append((example_only, labels))

    result_strs = []
    for example, labels in results:
        result_strs.extend(create_examples_by_tags(example, labels))
    save(destination_filepath, result_strs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="./data_learn/klue_ner_train_80.t")
    parser.add_argument("--test", type=str, default="./data_learn/klue_ner_test_20.t")
    parser.add_argument(
        "--dest_train",
        type=str,
        default="./data_learn/preprocessed/klue_ner_t2t_train.tsv",
    )
    parser.add_argument(
        "--dest_test",
        type=str,
        default="./data_learn/preprocessed/klue_ner_t2t_test.tsv",
    )

    args = parser.parse_args()

    print(f"전처리: {args.train} -> {args.dest_train}")
    preprocess_data_for_t2t(args.train, args.dest_train)
    print(f"전처리: {args.test} -> {args.dest_test}")
    preprocess_data_for_t2t(args.test, args.dest_test)
