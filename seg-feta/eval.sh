#!/bin/bash
#SBATCH -J BrainTumour
#SBATCH -A P00120220004
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
python eval_resunet.py
