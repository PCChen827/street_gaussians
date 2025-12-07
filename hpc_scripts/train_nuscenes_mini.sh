#!/bin/bash
#SBATCH --job-name=ns_mini
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --output=slurm_%j.out

# Load modules (if needed)
module purge
module load gcc/11.2.0
module load cuda/12.8.1

# Activate your conda environment
eval "$(conda shell.bash hook)"
conda activate street-gaussian

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ========= Start shell =========
# NuScenes mini 000
python ../train.py --config configs/nuscenes/nuscenes_mini_000.yaml

# NuScenes mini 003
python ../train.py --config configs/nuscenes/nuscenes_mini_003.yaml
