#!/bin/sh
echo $SLURM_JOB_ID >> jobs
{ 
echo   
python /private/home/xwhan/process_wiki/process_wikipedia.py 1
kill -9 $$
} & 
child_pid=$!
trap "echo 'TERM Signal received';" TERM
trap "echo 'Signal received'; if [ "$SLURM_PROCID" -eq "0" ]; then sbatch nsample_scripts/run.nsample_1.slrm; fi; kill -9 $child_pid; " USR1
while true; do     sleep 1; done
