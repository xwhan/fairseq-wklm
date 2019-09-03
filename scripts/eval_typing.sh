#!/bin/bash

thresh="0.4 0.5 0.6 0.7 0.8 0.9"

for t in $thresh
    do 
        python scripts/evaluate_typing.py --arch typing /private/home/xwhan/dataset/FIGER --model-path /checkpoint/xwhan/2019-08-31/typing_bert_base.typing.adam.lr5e-05.bert.seed3.bsz32.maxlen256.drop0.1.ngpu8/checkpoint_best.pt --use-sep --thresh $t;
done