#!/bin/bash
#SBATCH --job-name=finetune-qwen
#SBATCH --output=logs/finetune_qwen_prm_mean_qa.out
#SBATCH --error=logs/finetune_qwen_prm_mean_qa.err
#SBATCH --time=12:00:00  
#SBATCH --partition=vci_gpu_priority,gpu_batch,gpu,preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G 
#SBATCH --nodes=1 
#SBATCH --ntasks=1

conda activate 224r
cd /home/sanjay/224r

python train_commonqa_prm.py --num_epochs 5 --batch_size 5 --model_name Qwen/Qwen2.5-1.5B-Instruct --out_dir results/exp3_commonqa_qwen_prm_mean --reward_type normalized_lm