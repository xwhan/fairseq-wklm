#!/bin/sh

# for shard_id in $(seq 0 168)
#     do
#         sbatch ./scripts_pre/run.preprocess_$shard_id.preprocess.slrm &
# done



for shard_id in $(seq 0 49)
    do
        sbatch ./scripts_pre/run.process_$shard_id.preprocess.slrm &
done



