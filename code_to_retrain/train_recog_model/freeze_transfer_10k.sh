#!/bin/sh
#BSUB -q gpuv100
#BSUB -J freeze_transfer_10k
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o out_f10k.out
#BSUB -e error_f10k.err

module load cuda/11.6


source /zhome/a7/0/155527/Desktop/s204161/fagprojekt/venv/bin/activate


python3 trainer.py 

