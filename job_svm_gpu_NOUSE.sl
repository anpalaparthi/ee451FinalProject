#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH --partition=gpu 
#SBATCH --output=svm_gpu_NOUSE.out 
#SBATCH --error=svm_gpu_NOUSE.err
#SBATCH --gres=gpu:v100:1

./svm_gpu_NOUSE