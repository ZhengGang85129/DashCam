#!/bin/bash
batch_size=$1
learning_rate=$2
aug_types=$3
workspce="/eos/user/y/ykao/SWAN_projects/kaggle/DashCam" # FIXME
monitor_dir="/eos/user/y/ykao/www/kaggle/20250315" # FIXME

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate dashcam
cd ${workspce}

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

    # Only use --augmentation_types, no need for --use_augmentation
    python3 train.py --batch_size ${batch_size} --learning_rate ${learning_rate} --monitor_dir ${monitor_dir} --augmentation_types${aug_args}
fi
