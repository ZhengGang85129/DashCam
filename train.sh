#!/bin/bash
batch_size=$1
learning_rate=$2
aug_types=$3
aug_prob=$4 # Takes probability as a number (default: 0.25 if not provided)
workspce="/eos/user/y/ykao/SWAN_projects/kaggle/DashCam" # FIXME
monitor_dir="/eos/user/y/ykao/www/kaggle/20250315" # FIXME

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate dashcam
cd ${workspce}

# Set default probability if not provided
if [ -z "$aug_prob" ]; then
    aug_prob=0.25
fi

# Set up augmentation arguments
if [ -z "$aug_types" ]; then
    # No augmentation if aug_types is empty
    python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} --monitor_dir ${monitor_dir}
else
    # Convert string to array of args for compatibility with nargs='+'
    read -ra aug_array <<< "$aug_types"
    aug_args=""
    for type in "${aug_array[@]}"; do
        aug_args+=" $type"
    done

    # Use the augmentation probability parameter
    python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} --monitor_dir ${monitor_dir} \
        --augmentation_types${aug_args} --augmentation_prob ${aug_prob}
fi
