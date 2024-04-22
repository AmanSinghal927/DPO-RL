#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH --output=jupyter-notebook-%J.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

module load python/intel/3.8.6  # Updated to the correct module name
module load cuda/11.6.2
source activate dpo  # Make sure the 'dpo' conda environment is correctly set up

# Set up a tunnel for the port Jupyter will use
# This sets up a port forwarding from the compute node to the login node and from there to your local machine
XDG_RUNTIME_DIR=""

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
