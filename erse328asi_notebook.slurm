#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256GB 
#SBATCH --time=72:00:00 
#SBATCH --partition=batch 

conda activate erse328asi

# Get tunneling info 
export XDG_RUNTIME_DIR="" node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=6789
echo $node pinned to port $port 

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa 

# Run Jupyter 
jupyter-notebook --no-browser --port=${port} --port-retries=50 --ip=${node}
