#!/bin/sh
echo $SLURM_JOB_ID >> jobs
{ 
echo kdn_16  
nvidia-smi
python /private/home/xwhan/fairseq-py/scripts/evaluate_kdn_lama.py /private/home/xwhan/dataset/kdn/ --boundary-loss --rel-id 16
kill -9 $$
} & 
child_pid=$!
trap "echo 'TERM Signal received';" TERM
trap "echo 'Signal received'; if [ "$SLURM_PROCID" -eq "0" ]; then sbatch scripts_kdn/run.kdn_16.slrm; fi; kill -9 $child_pid; " USR1
while true; do     sleep 1; done
