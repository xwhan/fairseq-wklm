# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)


DICTIONARY_LOCATION="/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt"

# SPLITS="train valid"
# DATASET="triviaqa"
# TASK_DATA_FOLDER="/private/home/xwhan/dataset/$DATASET/processed-splits"
# OUT_DATA_FOLDER="/private/home/xwhan/dataset/$DATASET/binarized"

TASK_DATA_FOLDER="/private/home/xwhan/dataset/WebQ/processed-splits"
OUT_DATA_FOLDER="/private/home/xwhan/dataset/WebQ/binarized"

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

# for shard_id in $(seq 0 9)
#     do 
#       for INPUT_TYPE in $VAR
#         do
#           LANG="input$INPUT_TYPE"
#           python /private/home/xwhan/fairseq-py/preprocess.py \
#             --only-source \
#             --trainpref $TASK_DATA_FOLDER/train/${INPUT_TYPE}_${shard_id}.txt \
#             --workers 50 \
#             --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/$shard_id/ \
#             --srcdict $DICTIONARY_LOCATION \
#             --task span_qa;
#         done
#     done



# for INPUT_TYPE in $VAR
#     do
#       LANG="input$INPUT_TYPE"
#       python /private/home/xwhan/fairseq-py/preprocess.py \
#         --only-source \
#         --validpref $TASK_DATA_FOLDER/valid/${INPUT_TYPE}_0.txt \
#         --destdir $OUT_DATA_FOLDER/$INPUT_TYPE/0/ \
#         --workers 50 \
#         --srcdict $DICTIONARY_LOCATION \
#         --task span_qa;
# done