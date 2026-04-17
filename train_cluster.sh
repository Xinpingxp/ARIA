#!/bin/bash

# SUTD Mega Cluster Training Script
# Submit this script to the cluster using: sbatch train_cluster.sh

#SBATCH --job-name=ad_classification
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Load modules
module load python/3.8
module load cuda/11.4
module load cudnn/8.2

# Activate environment
source ~/miniconda3/bin/activate ad_env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
cd /path/to/your/project  # Replace with actual path
python src/train.py

echo "Training completed"