#!/bin/bash
#SBATCH -A cin_extern02
#SBATCH -p m100_usr_prod
#SBATCH --time 00:20:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2        # 2 gpus per node out of 4
#SBATCH --job-name=ddp

module load autoload profile/deeplrn
module load cineca-ai

srun ./scripts/train.sh "$@"
