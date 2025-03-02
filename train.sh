#!/bin/bash
batch_size=$1
learning_rate=$2
workspce="/eos/user/y/ykao/SWAN_projects/kaggle/DashCam" # FIXME

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate dashcam
cd ${workspce}
python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate}

#--------------------------------------------------
# commands to use customized directory
#--------------------------------------------------
# monitor_dir="/eos/user/y/ykao/www/kaggle/20250228"
# python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} --monitor_dir ${monitor_dir}
