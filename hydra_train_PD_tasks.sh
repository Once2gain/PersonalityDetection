for k in {0..9}
do
    FOLD="fold-$k"
    /root/anaconda3/envs/fairseq/bin/python3 /data/tangqirui/fairseq/hydra_train.py --config-dir "DATA-bin/kaggle-mbti/mbti-3-5/$FOLD/doc_dataset/config" --config-name "ei"
done

for k in {0..9}
do
    FOLD="fold-$k"
    /root/anaconda3/envs/fairseq/bin/python3 /data/tangqirui/fairseq/hydra_train.py --config-dir "DATA-bin/kaggle-mbti/mbti-3-5/$FOLD/doc_dataset/config" --config-name "jp"
done

for k in {0..9}
do
    FOLD="fold-$k"
    /root/anaconda3/envs/fairseq/bin/python3 /data/tangqirui/fairseq/hydra_train.py --config-dir "DATA-bin/kaggle-mbti/mbti-3-5/$FOLD/doc_dataset/config" --config-name "sn"
done

for k in {0..9}
do
    FOLD="fold-$k"
    /root/anaconda3/envs/fairseq/bin/python3 /data/tangqirui/fairseq/hydra_train.py --config-dir "DATA-bin/kaggle-mbti/mbti-3-5/$FOLD/doc_dataset/config" --config-name "tf"
done


