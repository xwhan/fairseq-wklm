#!/bin/bash

thresh="0.5 0.525 0.55 0.575 0.6 0.65"

for t in $thresh
    do 
        python scripts/evaluate_typing.py --arch typing /private/home/xwhan/dataset/FIGER --model-path /checkpoint/xwhan/2019-09-03/typing_bert_base.typing.adam.lr3e-05.bert.seed5.maxlen64.bsz64.drop0.1.ngpu16/checkpoint_best.pt --use-sep --thresh $t;
done