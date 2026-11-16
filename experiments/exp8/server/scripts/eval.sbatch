#!/bin/bash
# Exp 8 benchmark eval on one H200. Usage (from wsp/):
#   sbatch --job-name=subham_wsp_eval_base exp8/scripts/eval.sbatch --weights base --bench math500,gsm8k,ifeval --out exp8/results/base.json
#SBATCH --job-name=subham_wsp_eval
#SBATCH --output=exp8/logs/%j_eval.out
#SBATCH --error=exp8/logs/%j_eval.err
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
cd /path/to/wsp
export PATH=$PWD/venv/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=$PWD
echo "start $(date) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID args=$*"
venv/bin/python exp8/eval.py "$@"
rc=$?
echo "end $(date) exit=$rc"
exit $rc
