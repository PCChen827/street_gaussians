#!/bin/bash
#SBATCH --job-name=waymo
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
# Waymo scene 023
python ../train.py --config configs/waymo/waymo_scene_023.yaml

# Waymo scene 552
# python ../train.py --config configs/waymo/waymo_scene_552.yaml