#!/bin/bash

DICTIONARY_LOCATION="/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt"

TASK_DATA_FOLDER="/private/home/xwhan/dataset/tacred/processed-splits"
OUT_DATA_FOLDER="/private/home/xwhan/dataset/tacred/binarized"

VAR="train"
for INPUT_TYPE in $VAR
    do
      python /private/home/xwhan/fairseq-py/preprocess.py \
        --only-source \
        --trainpref $TASK_DATA_FOLDER/$INPUT_TYPE/sent.txt \
        --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/ \
        --workers 50 \
        --srcdict $DICTIONARY_LOCATION \
        --task kdn;
done


VAR="valid"
for INPUT_TYPE in $VAR
    do
      python /private/home/xwhan/fairseq-py/preprocess.py \
        --only-source \
        --validpref $TASK_DATA_FOLDER/$INPUT_TYPE/sent.txt \
        --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/ \
        --workers 50 \
        --srcdict $DICTIONARY_LOCATION \
        --task kdn;
done