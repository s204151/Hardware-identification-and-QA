#!/bin/sh
#BSUB -q gpua100
#BSUB -J split_data_cropped
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o out_spilts.out
#BSUB -e error_splits.err

module load cuda/11.6


source /zhome/a7/0/155527/Desktop/s204161/fagprojekt/venv_2/bin/activate


python3 split_data_for_holdout.py 

