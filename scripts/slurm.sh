#!/bin/bash
#SBATCH -A tra22_Nvaitc
#SBATCH -p m100_sys_test
#SBATCH -q qos_test
#SBATCH --time 01:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2        # n gpus per node out of 4
#SBATCH --job-name=ner
#SBATCH -o job.out          # for stdout redirection
#SBATCH -e job.err          # for stderr redirection

export WANDB_CACHE_DIR=$CINECA_SCRATCH/wandb_cache
export HF_DATASETS_CACHE=$CINECA_SCRATCH/hf_datasets_cache

srun ./scripts/train.sh "$@"
