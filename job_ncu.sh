#!/bin/bash
#SBATCH --job-name=pingpong_ncu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=0-00:04
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL,DISABLE_DCGM=1

mkdir -p logs

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
ADD_PATH="$SCRIPT_DIR/../../add_path.sh"

module load apptainer


apptainer exec --nv ~/apptainer.sandbox bash -c "
    source ${ADD_PATH} &&
    cd ${SCRIPT_DIR} &&
    ncu --set full -f -o logs/pingpong_gemm_${SLURM_JOB_ID} \
        python3.12 gemm_pingpong.py
"