#!/bin/bash
#SBATCH --job-name=finetune-llama
#SBATCH --output=logs/finetune_llama_prm_norm_qa.out
#SBATCH --error=logs/finetune_llama_prm_norm_qa.err
#SBATCH --time=12:00:00  
#SBATCH --partition=vci_gpu_priority,gpu_batch,gpu,preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G 
#SBATCH --nodes=1 
#SBATCH --ntasks=1

conda activate 224r
cd /home/sanjay/224r

python train_commonqa_prm.py --num_epochs 5 --batch_size 3 --model_name meta-llama/Llama-3.2-3B-Instruct --out_dir results/exp3_commonqa_llama_prm_norm --reward_type normalized_lm_variance