#!/bin/bash
nvidia-smi
source /path/to/your/conda/bin/activate # FIXME 
conda activate dashcam
cd YOURWORKSPACE # FIXME
python3 train.py
