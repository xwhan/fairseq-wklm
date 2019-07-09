# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
GLUE_DATA_FOLDER="/private/home/namangoyal/dataset/glue/data"
BPE_ENCODER_LOCATION="/private/home/namangoyal/src/gpt2_bpe"
# DICTIONARY_LOCATION="/private/home/myleott/data/data-bin/CC-NEWS-en.v5/dict.txt"
DICTIONARY_LOCATION="/private/home/myleott/data/data-bin/CC-NEWS-en.v8/dict.txt"

SPLITS="train dev test-middle test-high"
INPUT_COUNT=4
TASK_DATA_FOLDER="/private/home/jingfeidu/data/race/preprocessed-split"
OUT_DATA_FOLDER="/private/home/jingfeidu/data/race/binarized-v8-split"

VAR="1 2 3 4 context"

for INPUT_TYPE in $VAR
    do
      LANG="input$INPUT_TYPE"
      python /private/home/jingfeidu/fairseq-py-ft-control/preprocess.py \
        --only-source \
        --trainpref $TASK_DATA_FOLDER/train.$INPUT_TYPE \
        --validpref $TASK_DATA_FOLDER/dev.$INPUT_TYPE \
        --testpref $TASK_DATA_FOLDER/test-middle.$INPUT_TYPE,$TASK_DATA_FOLDER/test-high.$INPUT_TYPE \
        --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/ \
        --workers 10 \
        --srcdict $DICTIONARY_LOCATION;
done