#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=3:00:00
#SBATCH --partition=gpu 
#SBATCH --output=mlp2_gpu_many_blob400.out 
#SBATCH --error=mlp2_gpu_many_blob400.err
#SBATCH --gres=gpu:v100:1

./mlp2_gpu_many_blob