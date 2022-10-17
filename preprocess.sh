echo "기존 파일 지우기"
rm -r ./data_learn/preprocessed

echo "t2t를 위한 데이터 전처리"
python data/preprocess/preprocess_data_for_t2t.py

echo "완료~"