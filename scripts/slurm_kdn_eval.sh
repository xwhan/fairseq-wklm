FAIRSEQ=/private/home/xwhan/fairseq-py

JOBSCRIPTS=scripts_kdn
mkdir -p ${JOBSCRIPTS}

queue=learnfair
SHARDS="0 1 2 3 4 5 6 7 8 9 10 11"
for shard_id in $SHARDS
    do
    echo $shard_id
    SWEEP_NAME=kdn_${shard_id}
    SAVE_ROOT=/checkpoint/xwhan/${SWEEP_NAME}
    JNAME=${SWEEP_NAME}
    SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
    SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm

    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=/checkpoint/xwhan/jobs/${JNAME}.out" >> ${SLURM}
    echo "#SBATCH --error=/checkpoint/xwhan/jobs/${JNAME}.err" >> ${SLURM}
    echo "#SBATCH --mail-user=xwhan@fb.com" >> ${SLURM}
    echo "#SBATCH --mail-type=none" >> ${SLURM}
    echo "#SBATCH --partition=$queue" >> ${SLURM}
    echo "#SBATCH --signal=USR1@120" >> ${SLURM}
    echo "#SBATCH --mem=100000" >> ${SLURM}
    echo "#SBATCH --time=500" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "#SBATCH --gres=gpu:1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=50" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
    echo "{ " >> ${SCRIPT}
    echo "echo $SWEEP_NAME $BSZ " >> ${SCRIPT}
    echo "nvidia-smi" >> ${SCRIPT}
    echo python /private/home/xwhan/fairseq-py/scripts/evaluate_kdn.py ~/dataset/kdn/ --boundary-loss --rel-id $shard_id >> ${SCRIPT}
    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}
done


# shards=$(seq 0 11)
for shard_id in $SHARDS
    do
        sbatch ./scripts_kdn/run.kdn_$shard_id.slrm &
done