FAIRSEQ=/private/home/jingfeidu/fairseq-py

DICTIONARY_LOCATION="/private/home/xwhan/fairseq-py/vocab_dicts/dict.txt"

SPLITS="train valid"

DATASET="kdn"
TASK_DATA_FOLDER="/checkpoint/xwhan/wiki_data/processed-splits-v2"
OUT_DATA_FOLDER="/checkpoint/xwhan/wiki_data/binarized-v2"

DATE=20190807

JOBSCRIPTS=scripts_pre
# JOBSCRIPTS=scripts_ann
mkdir -p ${JOBSCRIPTS}

queue=learnfair
for shard_id in $(seq 0 9)
    do
    SWEEP_NAME=process_${shard_id}
    SAVE_ROOT=/checkpoint/xwhan/${DATE}/${SWEEP_NAME}
    mkdir -p /checkpoint/xwhan/stdout /checkpoint/xwhan/stdout
    JNAME=${SWEEP_NAME}.preprocess
    SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
    SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm

    extra=""
    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=/checkpoint/xwhan/stdout/${JNAME}.%j" >> ${SLURM}
    echo "#SBATCH --error=/checkpoint/xwhanstderr/${JNAME}.%j" >> ${SLURM}
    echo "#SBATCH --mail-user=xwhan@fb.com" >> ${SLURM}
    echo "#SBATCH --mail-type=none" >> ${SLURM}
    echo "#SBATCH --partition=$queue" >> ${SLURM}
    echo "#SBATCH --signal=USR1@120" >> ${SLURM}
    echo "#SBATCH --mem=1000000" >> ${SLURM}
    echo "#SBATCH --time=100" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=50" >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
    echo "{ " >> ${SCRIPT}
    echo "echo $SWEEP_NAME $BSZ " >> ${SCRIPT}
    echo "nvidia-smi" >> ${SCRIPT}
    echo python /private/home/xwhan/fairseq-py/preprocess.py \
        --only-source \
        --trainpref $TASK_DATA_FOLDER/train/context_$shard_id.txt \
        --destdir $OUT_DATA_FOLDER/train/shard_$shard_id/ \
        --workers 40 \
        --srcdict $DICTIONARY_LOCATION \
        --task kdn  >> ${SCRIPT}
    # echo python /private/home/xwhan/process_wiki/process_wikipedia.py $shard_id  >> ${SCRIPT}

    echo "nvidia-smi" >> ${SCRIPT}
    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}
done


for shard_id in $(seq 0 49)
    do
        sbatch ./scripts_pre/run.process_$shard_id.preprocess.slrm &
done