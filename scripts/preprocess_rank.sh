DICTIONARY_LOCATION="/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt"

SPLITS="train valid"

DATASET="triviaqa-ranking"
TASK_DATA_FOLDER="/private/home/xwhan/dataset/$DATASET/processed-splits"
OUT_DATA_FOLDER="/private/home/xwhan/dataset/$DATASET/binarized"

VAR="q c"

for INPUT_TYPE in $VAR
    do
      LANG="input$INPUT_TYPE"
      python /private/home/xwhan/fairseq-py/preprocess.py \
        --only-source \
        --trainpref $TASK_DATA_FOLDER/train/$INPUT_TYPE.txt \
        --validpref $TASK_DATA_FOLDER/valid/$INPUT_TYPE.txt \
        --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/ \
        --workers 50 \
        --srcdict $DICTIONARY_LOCATION \
        --task span_qa;
done