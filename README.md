# T5 for NER (Named Entity Recognition)

`KT GenieLabs Dev-Challenge 2022` <초거대T5 모델 기반 개체명 인식 파인 튜닝>에 본선 진출한 `ICU` 팀의 코드입니다 ✌✌

---------------------------------------

## Overview

[T5(T5: Text-To-Text Transfer Transformer)](https://github.com/google-research/text-to-text-transfer-transformer) 모델을 한국어 개체명 인식(NER) task를 위해 fine-tuning 및 test합니다. 

### Dataset 

한국어 NER task의 fine-tuning에 사용한 dataset은 [KLUE benchmark](https://github.com/KLUE-benchmark/KLUE) NER dataset입니다. 

### Pre-trained model 

본선 당시에는 제공받은 한국어 pre-trained T5 model을 사용하였지만, 본 repo에서는 multilingual T5 model을 사용합니다. 


## Environments

### Setting 

python 3.8 버전 및 GPU 환경을 권장합니다. 

### Requirements 

```
$ pip install -r requirements.txt
```

## Training 

```
$ fill_this_part
```


## Testing 

```
$ fill_this_part
```

## Scores 

`seed = 928` 에서의 hyperparamter tuning을 위한 grid-search 결과는 아래와 같습니다. (대회에서 제공받은 PLM을 사용한 결과입니다.)

| model size | max_len(input, output) | batch_size | num_epochs | optimizer | learning_rate | test f1 | 
| ----------|----------| ----------|----------|----------|----------|----------|
| small | 512, 128 | 12 | 10 | AdamW | 2e-5 | 0.8663 | 
| small | 128, 128 | 64 | 10 | AdamW | 4e-5 | **0.9171** | 
| base | 512, 128 | 12 | 5 | AdamW | 2e-5 | 0.9159 | 
| base | 512, 128 | 12 | 10 | AdamW | 2e-5 | 0.9119 | 
| small | 128, 128 | 64 | 10 | AdamW | 3e-5 | 0.9107 | 
| small | 256, 128 | 64 | 5 | AdamW | 3e-5 | 0.9068 | 
| small | 128, 128 | 64 | 5 | AdamW | 4e-5 | 0.9044 | 
| small | 128, 128 | 64 | 10 | AdamW | 2e-5 | 0.9030 | 
| small | 64,64 | 64 | 10 | AdamW | 2e-5 | 0.9004 | 
| small | 256, 128 | 32 | 10 | AdamW | 2e-5 | 0.8980 | 
| small | 512, 128 | 32 | 10 | AdamW | 2e-5 | 0.8980 | 
| small | 256, 128 | 32 | 10 | AdamW | 4e-5 | 0.8971 | 


## Developers 

- [허태정](https://github.com/Aqudi) from Yonsei Univ., Master's course in [Internet Computing Lab](http://icl.yonsei.ac.kr/).

- [심미단](https://github.com/midannii) from Yonsei Univ., Master's course in [Internet Computing Lab](http://icl.yonsei.ac.kr/).

- [공예슬](https://github.com/0ys) from Yonsei Univ., Master's course in [Internet Computing Lab](http://icl.yonsei.ac.kr/).
