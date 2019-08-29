

DICTIONARY_LOCATION="/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt"

SPLITS="train valid"

DATASET="kdn"
TASK_DATA_FOLDER="/checkpoint/xwhan/wiki_data_mlm/processed-splits"
OUT_DATA_FOLDER="/checkpoint/xwhan/wiki_data_mlm/binarized"

for shard_id in $(seq 0 1)
    do
      python /private/home/xwhan/fairseq-py/preprocess.py \
        --only-source \
        --trainpref $TASK_DATA_FOLDER/train/context_$shard_id.txt \
        --destdir $OUT_DATA_FOLDER/train/shard_$shard_id/ \
        --workers 50 \
        --srcdict $DICTIONARY_LOCATION \
        --task kdn;
done


VAR="valid"
for INPUT_TYPE in $VAR
    do
      python /private/home/xwhan/fairseq-py/preprocess.py \
        --only-source \
        --validpref $TASK_DATA_FOLDER/valid/context_0.txt \
        --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/shard_0/ \
        --workers 50 \
        --srcdict $DICTIONARY_LOCATION \
        --task kdn;
done