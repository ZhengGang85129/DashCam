#!/bin/bash
batch_size=$1
learning_rate=$2
aug_mode=$3
workspce="/eos/user/y/ykao/SWAN_projects/kaggle/DashCam" # FIXME

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate dashcam
cd ${workspce}

# Set up augmentation arguments based on aug_mode
# 0: No augmentation
# 1: Basic augmentation only
# 2: Basic + advanced augmentation
aug_args=""
if [ "$aug_mode" -eq "1" ]; then
    aug_args="--use_augmentation"
elif [ "$aug_mode" -eq "2" ]; then
    aug_args="--use_augmentation --use_advanced_augmentation"
fi

# Run training with the specified parameters
python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} ${aug_args}

#--------------------------------------------------
# commands to use customized directory
#--------------------------------------------------
# monitor_dir="/eos/user/y/ykao/www/kaggle/20250228"
# python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} --monitor_dir ${monitor_dir}
