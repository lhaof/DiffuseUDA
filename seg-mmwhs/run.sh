#!/bin/bash
#SBATCH -A F00120230017
#SBATCH -J test_nnUNet          # 作业名是 test
#SBATCH -p p-A100
#SBATCH -N 1                 # 使用2个节点
#SBATCH --ntasks=1  # 每个节点开启6个进程

#SBATCH --cpus-per-task=6 #如果是gpu任务需要在此行定义gpu数量,此处为1
#SBATCH --gres=gpu:1  
# module load cuda11.6/toolkit/11.6.1
nvidia-smi
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate genericssl
export PYTHONPATH=$(pwd)/code:$PYTHONPATH
# CUDA_LAUNCH_BLOCKING=1 python ./code/data/preprocess_mmwhs.py
# bash train.sh -c 0 -e diffusion -t mmwhs_ct2mr_g -i '1' -l 2e-3 -w 10 -n 150 -d false 
# bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_asc -i '2' -l 1e-2 -w 10 -n 150 -d true 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_asc -i '3' -l 5e-3 -w 10 -n 200 -d true 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_asc_infer -i '4' -l 2e-3 -w 10 -n 120 -d true 

# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_asc_pl -i '4' -l 2e-3 -w 10 -n 120 -d false 
# bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_asc_pl -i '4' -l 2e-3 -w 10 -n 120 -d false 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_diff_uda -i '1' -l 2e-3 -w 10 -n 30 -d true 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_diff_uda -i '2' -l 2e-3 -w 10 -n 30 -d true 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_diff_uda -i '3' -l 2e-3 -w 10 -n 30 -d true 
bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_diff_uda -i '11' -l 2e-3 -w 10 -n 30 -d true 
bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_diff_uda -i '12' -l 2e-3 -w 10 -n 30 -d true 
bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_diff_uda -i '13' -l 2e-3 -w 10 -n 30 -d true 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_diff_uda -i '4' -l 2e-3 -w 10 -n 45 -d false 
# bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_diff_uda -i '4' -l 2e-3 -w 10 -n 45 -d false 
# bash train.sh -c 0 -e asc_crop -t mmwhs_mr2ct_diff_uda -i '5' -l 2e-3 -w 10 -n 45 -d false 
# bash train.sh -c 0 -e asc_crop -t mmwhs_ct2mr_diff_uda -i '5' -l 2e-3 -w 10 -n 45 -d false 
# bash train.sh -c 0 -e diffusion -t mmwhs_ct_UPPER -i '1' -l 2e-3 -w 10 -n 180 -d true 
# bash train.sh -c 0 -e diffusion -t mmwhs_ct_UPPER -i '3' -l 5e-3 -w 10 -n 180 -d true 
# bash train.sh -c 0 -e diffusion -t mmwhs_mr_UPPER -i '3' -l 5e-3 -w 10 -n 180 -d true 

