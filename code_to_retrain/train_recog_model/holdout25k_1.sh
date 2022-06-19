#!/bin/sh
#BSUB -q gpuv100
#BSUB -J holdout
#BSUB -W 7:00
#BSUB -n 8
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o out_holdout.out
#BSUB -e error_holdout.err

module load cuda/11.6

nvidia-smi

source /zhome/a7/0/155527/Desktop/s204161/fagprojekt/venv/bin/activate

python3 trainer.py 

