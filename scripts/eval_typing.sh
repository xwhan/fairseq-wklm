#!/bin/bash

thresh="0.5 0.525 0.55 0.575 0.6 0.65"

#thresh="0.6"

for t in $thresh
    do 
        python scripts/evaluate_typing.py --arch typing /private/home/xwhan/dataset/FIGER --model-path /checkpoint/xwhan/2019-09-13/figer_mask0.15.typing.adam.lr2e-05.kdn_v2_boundary.seed3.maxsent16.maxlen256.drop0.1.ngpu16/checkpoint_best.pt    --eval-data /private/home/xwhan/dataset/FIGER/processed-splits/test  --use-kdn --boundary-loss --use-marker --thresh $t;
done
