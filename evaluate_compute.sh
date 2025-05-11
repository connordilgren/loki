#!/bin/bash
#SBATCH --job-name=loki_evaluate_compute      # Job name
#SBATCH --output=loki_evaluate_compute_%j.out # Output file (%j expands to jobID)
#SBATCH --error=loki_evaluate_compute_%j.err  # Error file (%j expands to jobID)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=32G                  # Memory requirement
#SBATCH --time=00:04:00            # Time limit (HH:MM:SS)
#SBATCH --partition=gpu            # Partition/queue name
#SBATCH --account=cmsc828-class
#SBATCH --gpus=a100:1          # Request specific GPU type
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cdilgren@umd.edu

# Load necessary modules (you may need to adjust these for your cluster)
module purge
module load cuda/12.3.0/gcc/11.3.0/x86_64

# Activate virtual environment
source /scratch/zt1/project/cmsc828/user/cdilgren/loki/.venv/bin/activate

# set triton cache dir
export TRITON_CACHE_DIR=/scratch/zt1/project/cmsc828/user/cdilgren/loki/triton_cache

# Run the Python script with your arguments
python evaluate_compute.py

# End of script
