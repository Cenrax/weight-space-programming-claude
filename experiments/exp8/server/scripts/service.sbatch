#!/bin/bash
# Exp 8 practice/eval HTTP service on one H200, listening on 127.0.0.1:${EXP8_PORT:-8800} of gpu-node.
# Start/stop with exp8/scripts/service_ctl.sh (idempotent; safe to re-run after the job ends).
#SBATCH --job-name=subham_wsp_service
#SBATCH --output=exp8/logs/%j_service.out
#SBATCH --error=exp8/logs/%j_service.err
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
cd /path/to/wsp
export PATH=$PWD/venv/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=$PWD
echo "start $(date) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID args=$*"
venv/bin/python exp8/service.py --port ${EXP8_PORT:-8800} "$@"
rc=$?
echo "end $(date) exit=$rc"
exit $rc
