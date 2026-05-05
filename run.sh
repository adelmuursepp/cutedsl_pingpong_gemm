#!/usr/bin/env bash
#!/bin/bash
#SBATCH --job-name=pingpong_gemm_run
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=0-00:30
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR" 
ADD_PATH="$SCRIPT_DIR/../../add_path.sh"

module load apptainer

apptainer exec --nv ~/apptainer.sandbox bash -c "source ${ADD_PATH} &&
 cd ${SCRIPT_DIR} &&
 python3.12 benchmark.py 4096 4096 256 "
