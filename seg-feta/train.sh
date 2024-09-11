#!/bin/bash
#SBATCH -J haifangong 
#SBATCH -A P00120220004 
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
python train_asc_v2.py --type '_stmx4' --train_data_s 'train_diff4.list'
# python train_resunet.py --type '_s'
# python train_resunet.py --type '_t'
# python train_resunet.py --type '_st' --train_data_s 'train diff two times.list'
# python train_resunet.py --type '_stm' --train_data_s 'train diff three times.list'
# python train_asc.py --type '_stm' --train_data_s 'train diff three times.list'