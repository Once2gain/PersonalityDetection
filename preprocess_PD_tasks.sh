#!/bin/bash

#ROOT="/data/tangqirui/fairseq"
SRC_DATA_DIR="DATA-bin/kaggle-mbti/mbti-3-5"
DEST_DATA_DIR="DATA-bin/kaggle-mbti/mbti-3-5"

DATASETS="seg_dataset doc_dataset"
for DATASET in $DATASETS
do
  echo "Preprocessing $DATASET"

  # Run fairseq preprocessing:
  for k in {0..9}
  do
    FOLD="fold-$k"
    python preprocess.py \
      --only-source \
      --trainpref "$SRC_DATA_DIR/$FOLD/$DATASET/train.input" \
      --validpref "$SRC_DATA_DIR/$FOLD/$DATASET/valid.input" \
      --destdir "$DEST_DATA_DIR/$FOLD/$DATASET/input" \
      --workers 60 \
      --srcdict "gpt2_bpe/dict.txt" \
      --enc-document;

    python preprocess.py \
      --only-source \
      --trainpref "$SRC_DATA_DIR/$FOLD/$DATASET/train.label" \
      --validpref "$SRC_DATA_DIR/$FOLD/$DATASET/valid.label" \
      --destdir "$DEST_DATA_DIR/$FOLD/$DATASET/label" \
      --workers 60;

#    cp -r "$SRC_DATA_DIR/$FOLD/$DATASET/feats" "$DEST_DATA_DIR/$FOLD/$DATASET"
#    cp -r "$TMP_DATA_DIR/$FOLD/$DATASET/config" "$DEST_DATA_DIR/$FOLD/$DATASET"
  done
done