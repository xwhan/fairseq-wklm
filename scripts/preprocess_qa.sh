# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)


DICTIONARY_LOCATION="/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt"

SPLITS="train valid"

DATASET="webq_qa"
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
        --workers 10 \
        --srcdict $DICTIONARY_LOCATION \
        --task span_qa;
done