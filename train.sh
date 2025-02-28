#!/bin/bash
batch_size=$1
learning_rate=$2

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate dashcam
cd /eos/user/y/ykao/SWAN_projects/kaggle/DashCam/ # FIXME
python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate}
