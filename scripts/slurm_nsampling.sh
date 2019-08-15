

JOBSCRIPTS=nsample_scripts
mkdir -p ${JOBSCRIPTS}

queue=learnfair

for shard_id in $(seq 0 9)
    do
    JNAME=nsample_${shard_id}
    mkdir -p /checkpoint/xwhan/stdout /checkpoint/xwhan/stderr
    SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
    SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm

    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=/checkpoint/xwhan/stdout/${JNAME}.%j" >> ${SLURM}
    echo "#SBATCH --error=/checkpoint/xwhan/stderr/${JNAME}.%j" >> ${SLURM}
    echo "#SBATCH --mail-user=xwhan@fb.com" >> ${SLURM}
    echo "#SBATCH --mail-type=none" >> ${SLURM}
    echo "#SBATCH --partition=$queue" >> ${SLURM}
    echo "#SBATCH --signal=USR1@120" >> ${SLURM}
    echo "#SBATCH --mem=500000" >> ${SLURM}
    echo "#SBATCH --time=800" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=64" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
    echo "{ " >> ${SCRIPT}
    echo "echo $SWEEP_NAME $BSZ " >> ${SCRIPT}

    echo python /private/home/xwhan/process_wiki/process_wikipedia.py $shard_id  >> ${SCRIPT}

    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}
done

for shard_id in $(seq 0 9)
    do
        sbatch ./nsample_scripts/run.nsample_$shard_id.slrm &
done