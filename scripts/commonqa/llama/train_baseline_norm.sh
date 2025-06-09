#!/bin/bash
#SBATCH --job-name=finetune-llama
#SBATCH --output=logs/finetune_llama_norm_qa.out
#SBATCH --error=logs/finetune_llama_norm_qa.err
#SBATCH --time=12:00:00  
#SBATCH --partition=vci_gpu_priority,gpu_batch,gpu,preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --nodes=1 
#SBATCH --ntasks=1

conda activate 224r
cd /home/sanjay/224r

python train_commonqa_baseline.py --num_epochs 5 --batch_size 3 --model_name meta-llama/Llama-3.2-3B-Instruct --out_dir results/exp2_commonqa_llama_baseline_norm --reward_type normalized_binary_variance --lr 5e-7