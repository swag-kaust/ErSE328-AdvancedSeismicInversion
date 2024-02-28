#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256GB 
#SBATCH --time=72:00:00 
#SBATCH --partition=batch 

# Load environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

# source ~/miniconda3/envs/my_env_plot/bin/activate
conda activate hcpinnseikonal

##########

# get tunneling info 
export XDG_RUNTIME_DIR="" node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=6789
echo $node pinned to port $port 
# print tunneling instructions 

echo -e " 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa 

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop " 

# Run Jupyter 
# jupyter notebook --no-browser --port=${port} --port-retries=50 --ip=${node} 
jupyter-lab --no-browser --port=${port} --port-retries=50 --ip=${node} 
