#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH --partition=gpu 
#SBATCH --output=mlp2_gpu_titanic.out 
#SBATCH --error=mlp2_gpu_titanic.err
#SBATCH --gres=gpu:v100:1

./mlp2_gpu_titanic