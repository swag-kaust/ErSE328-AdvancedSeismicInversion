#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh

# Create conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate erse328asi
conda env list
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
echo 'Created and activated environment:' $(which python)

# Check cupy works as expected
echo 'Checking torch version and GPU'
conda activate erse328asi
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'
echo 'Done!'