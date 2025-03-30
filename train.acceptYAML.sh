#!/bin/bash
yaml_file=$1
workspce=${2-$PWD} # if not specified, use current directory

if [ "$yaml_file" == "" ]; then echo "[YamlRequired] You should input a yaml file to activate this training."; exit; fi

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate dashcam
cd ${workspce}

python3 train.py $yaml_file
