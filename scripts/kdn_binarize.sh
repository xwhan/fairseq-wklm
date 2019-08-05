#!/bin/sh

for shard_id in $(seq 0 99)
    do
        sbatch ./scripts_pre/run.preprocess_$shard_id.preprocess.slrm &
done


