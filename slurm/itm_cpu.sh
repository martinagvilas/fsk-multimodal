#!/bin/bash
#SBATCH --job-name=itm_vilt
#SBATCH --begin=now
#SBATCH --partition=8GBL
##SBATCH --gpus=rtxa6000:1
#SBATCH --output=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.out
#SBATCH --error=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=martinagonzalezvilas@gmail.com
#SBATCH --array=[0-342]

module purge
module load conda/4.10.1
conda activate fsk
cd /mnt/hpc/home/gonzalezvm/fsk-multimodal/fsk/it_match/
srun python run.py -m vilt -pp /mnt/hpc/home/gonzalezvm/fsk-multimodal/ -bi $SLURM_ARRAY_TASK_ID 