#!/bin/bash
#SBATCH --job-name=finetune-qwen
#SBATCH --output=logs/%x_%j.out        # Save stdout to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err         # Save stderr to logs/jobname_jobid.err
#SBATCH --time=24:00:00                # Max run time (hh:mm:ss)
#SBATCH --partition=gpu                # Cluster partition (adjust to your system)
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32G                      # RAM
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)

# Optional: Load modules or activate environment
module load anaconda
source activate your-conda-env  # or use `conda activate`

# Optional: Change directory to where your code is
cd /path/to/your/project

# Run your Python script
python train_qwen.py --epochs 5 --batch_size 4 --train_size 5000