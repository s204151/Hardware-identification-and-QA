#!/bin/sh
#BSUB -q gpuv100
#BSUB -J crafty
#BSUB -W 8:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o out_10kcraft.out
#BSUB -e error_10kcraft.err

module load cuda/11.6

nvidia-smi

source /zhome/a7/0/155527/Desktop/s204161/fagprojekt/venv_2/bin/activate


python3 works_trainSynth.py 

