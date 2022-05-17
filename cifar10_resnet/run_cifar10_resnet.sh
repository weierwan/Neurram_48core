#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH -G 1

ml load py-tensorflow/2.1.0_py36
ml load py-keras/2.3.1_py36

srun python cifar10_resnet.py