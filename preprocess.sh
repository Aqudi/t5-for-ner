echo "기존 파일 지우기"
rm -r ./data_learn/preprocessed

echo "t2t를 위한 데이터 전처리"
PWD=$(pwd)
python $PWD/data/preprocess_data_for_t2t.py\
    --train=$PWD/data_learn/klue_ner_train_80.t\
    --test=$PWD/data_learn/klue_ner_test_20.t\
    --dest_train=$PWD/data_learn/preprocessed/klue_ner_t2t_train.tsv\
    --dest_test=$PWD/data_learn/preprocessed/klue_ner_t2t_test.tsv


echo "완료~"