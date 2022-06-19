#!/bin/sh
#BSUB -q gpuv100
#BSUB -J holdout_vgg
#BSUB -W 12:00
#BSUB -n 8
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o out_vgg_holdout.out
#BSUB -e error_vgg_holdout.err

module load cuda/11.6

nvidia-smi

source /zhome/a7/0/155527/Desktop/s204161/fagprojekt/venv/bin/activate

python3 trainer_vgg.py 

