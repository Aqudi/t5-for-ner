# 나머지 import
import argparse

import pytorch_lightning as pl
from setproctitle import setproctitle

from datamodule import T5NerFineTunerDataModule
from evaluation_config import args_dict
from evaluation_module import T5NerFineTuner

setproctitle("evaluate-t5-ner")

args = argparse.Namespace(**args_dict)

# %%
trainer = pl.Trainer.from_argparse_args(args)
datamodule = T5NerFineTunerDataModule(**args_dict)

# %%
datamodule.prepare_data()
datamodule.setup()

# %%
total_steps = len(datamodule.train_dataloader())
model = T5NerFineTuner(total_steps=total_steps, **args_dict)


# %%
results = trainer.predict(model, dataloaders=datamodule.test_dataloader())


# %%
from tqdm.auto import tqdm
from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_name_or_path, fast=True)

predicted_tags = []
for i, r in enumerate(tqdm(results)):
    decoded_strings = tokenizer.decode(r, skip_special_tokens=True)
    predicted_tags.extend(
        list(
            map(
                lambda x: x.strip(),
                filter(lambda x: x != "", decoded_strings.split("[label] ")),
            )
        )
    )


# %%
sentences = []
labels = []
test_dataloader = datamodule.test_dataloader()

for i, r in enumerate(tqdm(test_dataloader)):
    batch_size = len(r["input_ids"])
    sentences.extend(
        [tokenizer.decode(t, skip_special_tokens=True) for t in r["input_ids"]]
    )
    labels.extend([tokenizer.decode(t, skip_special_tokens=True) for t in r["labels"]])


# %%
# remove [sentence], [label] indicators
sentence_start = len("[sentence]")
label_start = len("[label]")
sentences = list(map(lambda x: x[sentence_start:].strip(), sentences))
labels = list(map(lambda x: x[label_start:].strip(), labels))

# %%
split_6_tags = []
split_6_sentences = []
split_6_labels = []
for i in range(len(predicted_tags)):
    if i % 6 == 0:
        split_6_tags.append([])
        split_6_sentences.append([])
        split_6_labels.append([])
    split_6_tags[-1].append(predicted_tags[i])
    split_6_sentences[-1].append(sentences[i])
    split_6_labels[-1].append(labels[i])

# %%
def strip_and_remove_last_comma(text):
    text = text.strip()
    if text.endswith(","):
        text = text[:-1]
    return text


def compute_f1_score(predicted_tags, labels, sentences, debug=False):
    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    def tqdm_zip(*args):
        zipped = zip(*args)
        if debug:
            return zipped
        return tqdm(zipped)

    ner_tags = ["QT", "PS", "LC", "DT", "TI", "OG"]
    tp_fp_fn = []
    for current_predicted_tags, current_labels, current_sentences in tqdm_zip(
        predicted_tags, labels, sentences
    ):
        tp, fp, fn = 0, 0, 0
        # 한 문장에 대한 f1 스코어 계산
        for idx, (ner_tag, predicted, label) in enumerate(
            zip(ner_tags, current_predicted_tags, current_labels)
        ):
            p_words = list(
                map(strip_and_remove_last_comma, predicted.split(f"{ner_tag}:"))
            )
            l_words = list(map(strip_and_remove_last_comma, label.split(f"{ner_tag}:")))
            debug_print("ner_tag:", ner_tag, "p_words:", p_words)
            debug_print("ner_tag:", ner_tag, "l_words:", l_words)

            # entity가 없는데 있다고 한 경우
            if l_words[0] == "O" and p_words[0] != "O":
                debug_print("0개, N개", end=" ")
                debug_print("fp:", fp, end=" -> ")
                fp += len(p_words[1:])
                debug_print(fp)
                continue

            # entity가 있는데 없다고 한 경우
            elif l_words[0] != "O" and p_words[0] == "O":
                debug_print("N개, 0개", end=" ")
                debug_print("fn:", fn, end=" -> ")
                fn += len(l_words[1:])
                debug_print(fn)
                continue

            # 둘 다 O인 경우에는 패스
            elif l_words[0] == "O" and p_words[0] == "O":
                debug_print("0개, 0개 pass~")
                continue

            # 둘 다 O가 아니므로 첫 칸은 빈칸 -> 제거
            p_words = p_words[1:]
            l_words = l_words[1:]

            # TP 계산
            temp_l_words = l_words[:]
            temp_p_words = p_words[:]
            for l_word in temp_l_words:
                find_word = False
                for p_word in temp_p_words[:]:
                    # label과 predicted가 일치한 경우
                    # predicted words list에서 해당 아이템
                    if l_word == p_word:
                        debug_print(f"<{l_word}>, <{p_word}>같음! tp:", tp, end=" -> ")
                        tp += 1
                        debug_print(tp)
                        find_word = True

                    # 예측: <서울시 강남구>, 정답: <서울시>인 경우
                    # 예측된 단어가 정답으로 시작할 때
                    #    -> 아닌 경우 <남서울시>
                    elif p_word.startswith(l_word):
                        debug_print(
                            f"<{l_word}> <{p_word}> p_word가 l_word로 시작!", tp, end=" -> "
                        )
                        tp += 1
                        debug_print(tp)
                        find_word = True
                        # === 이후 상황

                        # 다음 l_word <강남구>로 이동
                        # p_word가 지워졌기 때문에 다음 l_word인 강남구는
                        # find_word = False로 자연스럽게 fn으로 처리됨

                    # 예측: <서울시>, 정답: <서울시 강남구>인 경우
                    # 정답이 예측된 단어로 시작될 때
                    #    -> 아닌 경우 <서울시청>
                    elif l_word.startswith(p_word):
                        debug_print(
                            f"<{l_word}> <{p_word}> l_word가 p_word로 시작!", tp, end=" -> "
                        )
                        tp += 1
                        find_word = True

                        # === 이후 상황

                        # 다음 l_word <대치동>로 이동
                        # 자연스럽게 다음 체크 과정 거침

                        # 강남구는 마지막까지 temp_p_words에서 사라지지 않아서
                        # named entity로 예측했지만 False인 FP로 처리됨

                    # 사용한 단어는 제거
                    # 다음 l_word 정답 체크
                    if find_word:
                        debug_print(f"단어 찾았으니까 <{p_word}> 지움!")
                        temp_p_words.remove(p_word)
                        break

                # p_words를 다 체크했는데 word를 찾지 못한 경우
                if not find_word:
                    debug_print(f"<{l_word}>가 predicted words에 없음! fn:", fn, end=" -> ")
                    fn += 1
                    debug_print(fn)

            # 마지막까지 label과 매칭되지 못한 예측된 NE들은
            # FP로 처리됨
            debug_print(temp_p_words, "가 label과 매칭되지 못했음 fp:", fp, end=" -> ")
            fp += len(temp_p_words)
            debug_print(fp)

        debug_print(current_sentences[0].split("[tag]")[0])
        debug_print("tp:", tp, "fp:", fp, "fn:", fn)
        debug_print()
        tp_fp_fn.append((tp, fp, fn))

    tp = sum([tff[0] for tff in tp_fp_fn])
    fp = sum([tff[1] for tff in tp_fp_fn])
    fn = sum([tff[2] for tff in tp_fp_fn])
    if (tp + fp) == 0:
        precision = 0
        print("### precision is zero ")
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
        print("### recall is zero ")
    else:
        recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# %%
results = compute_f1_score(
    split_6_tags,
    split_6_labels,
    split_6_sentences,
    debug=args.debug,
)
with open(
    "results/evaluation_results/f1_score_results.txt", "at", encoding="utf8"
) as f:
    import datetime

    f.write(f"{datetime.datetime.now()}\n")
    f.write(f"{results}\n")
    f.write(f"config: {args_dict}\n\n")


# %%
###################
# Test cases
###################
# test_tags = [
#     ["O", "O", "LC:서울시,LC:강남구", "O", "O", "O"],
#     ["O", "O", "LC:서울시 강남구", "O", "O", "O"],
#     ["O", "O", "LC:서울시,LC:강남구", "O", "O", "O"],
#     ["O", "PS:허태정", "O", "O", "O", "O"],
#     ["O", "PS:허태정", "O", "O", "O", "O"],
#     ["O", "O", "O", "O", "O", "O"],
#     ["O", "PS:허태정", "O", "O", "O", "O"],
#     ["O", "O", "O", "O", "O", "O"],
# ]
# test_labels = [
#     ["O", "O", "LC:서울시,LC:강남구", "O", "O", "O"],
#     ["O", "O", "LC:서울시,LC:강남구", "O", "O", "O"],
#     ["O", "O", "LC:서울시 강남구", "O", "O", "O"],
#     ["O", "PS:감자튀김", "O", "O", "O", "O"],
#     ["O", "O", "LC:허태정", "O", "O", "O"],
#     ["O", "O", "O", "O", "O", "O"],
#     ["O", "O", "O", "O", "O", "O"],
#     ["O", "PS:허태정", "O", "O", "O", "O"],
# ]
# test_sentences = [["허태정" + str(j) for j in range(6)] for i in range(len(test_labels))]
# f1_scores = compute_f1_score(test_tags, test_labels, test_sentences, debug=True)
# print(f1_scores)
