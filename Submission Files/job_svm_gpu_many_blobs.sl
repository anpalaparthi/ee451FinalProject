#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=0:20:00
#SBATCH --partition=gpu 
#SBATCH --output=svm_gpu_many_blob.out 
#SBATCH --error=svm_gpu_many_blob.err
#SBATCH --gres=gpu:v100:1

./svm_gpu_many_blob