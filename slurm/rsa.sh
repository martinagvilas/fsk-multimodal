#!/bin/bash
#SBATCH --job-name=rsa
#SBATCH --begin=now
#SBATCH --partition=16GBL
#SBATCH --output=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.out
#SBATCH --error=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=martinagonzalezvilas@gmail.com
#SBATCH --array=[17]
##SBATCH --array=[0-26]

module purge
module load conda/4.10.1
conda activate fsk
cd /mnt/hpc/home/gonzalezvm/fsk-multimodal/fsk/similarity/
srun python run.py -all -bi $SLURM_ARRAY_TASK_ID -pp /mnt/hpc/home/gonzalezvm/fsk-multimodal/ 