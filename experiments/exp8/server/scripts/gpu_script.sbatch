#!/bin/bash
# Run one exp8 python script on one H200. Usage (from wsp/): sbatch exp8/scripts/gpu_script.sbatch exp8/scripts/swap_test.py
#SBATCH --job-name=subham_wsp_script
#SBATCH --output=exp8/logs/%j_script.out
#SBATCH --error=exp8/logs/%j_script.err
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
cd /path/to/wsp
export PATH=$PWD/venv/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=$PWD
echo "start $(date) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID args=$*"
venv/bin/python "$@"
echo "end $(date) exit=$?"
