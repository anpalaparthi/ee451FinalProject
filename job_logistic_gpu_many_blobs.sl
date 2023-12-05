#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH --partition=gpu 
#SBATCH --output=logistic_gpu_many_blob.out 
#SBATCH --error=logistic_gpu_many_blob.err
#SBATCH --gres=gpu:v100:1

./logistic_gpu_many_blob