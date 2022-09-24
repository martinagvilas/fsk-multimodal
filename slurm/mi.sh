#!/bin/bash
#SBATCH --job-name=uni
#SBATCH --begin=now
#SBATCH --partition=16GBL
#SBATCH --output=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.out
#SBATCH --error=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=martinagonzalezvilas@gmail.com

module purge
module load conda/4.10.1
conda activate fsk
cd /mnt/hpc/home/gonzalezvm/fsk-multimodal/fsk/feature_repr
srun python mutual_information.py -m clip -pp /mnt/hpc/home/gonzalezvm/fsk-multimodal/