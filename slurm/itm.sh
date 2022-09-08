#!/bin/bash
#SBATCH --job-name=itm
#SBATCH --begin=now
#SBATCH --partition=GPUtest
#SBATCH --gpus=1
#SBATCH --output=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.out
#SBATCH --error=/mnt/hpc/home/gonzalezvm/slurm/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=martinagonzalezvilas@gmail.com

module purge
module load conda/4.10.1
conda activate fsk
cd /mnt/hpc/home/gonzalezvm/multimodal-fsk/fsk/it_match/
srun python run.py -m albef -pp /mnt/hpc/home/gonzalezvm/fsk-multimodal/