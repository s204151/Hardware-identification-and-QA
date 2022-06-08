#!/bin/sh
#BSUB -q gpuv100
#BSUB -J freeze_transfer_30k
#BSUB -W 0:03
#BSUB -n 8
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o out_wandb1.out
#BSUB -e error_wandb1.err

module load cuda/11.6


source /zhome/a7/0/155527/Desktop/s204161/fagprojekt/venv/bin/activate


python3 trainer.py 

