#!/bin/bash
#SBATCH -p short
#SBATCH --mem=3g
#SBATCH -o /dev/null

## to use this script
## sbatch -a 1-$(cat commands.list|wc -l) array_jobs.sh

CMD=$(head -n $SLURM_ARRAY_TASK_ID commands.list | tail -1)
echo ${CMD} | bash
