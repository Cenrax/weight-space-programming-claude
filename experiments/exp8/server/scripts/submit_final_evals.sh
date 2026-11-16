#!/bin/bash
# Exp 8 phase 2: all final evaluations as separate SLURM eval jobs, run one at a time on the single GPU allowed per
# user (the association limit queues them automatically; --dependency keeps the order explicit).
# Run on the login shell AFTER the practice service is stopped (never next to a Claude run, playbook step 22).
#   exp8/scripts/submit_final_evals.sh <gemma_edit_id> <b10_id> <b25_id> <b50_id> [gemma_scale]
set -e
cd /path/to/wsp
G=$1; B10=$2; B25=$3; B50=$4; GS=${5:-}
S="math500=4,gsm8k=2,ifeval=2"
sub() { sbatch --parsable ${DEP:+--dependency=afterany:$DEP} "$@"; }
DEP=""
# Gemma control (MATH-500 x4, 8k tokens), a few minutes
J=$(sub --job-name=subham_wsp_ev_gemma --time=03:00:00 --export=ALL,EXP8_FAMILY=gemma exp8/scripts/eval.sbatch \
    --model gemma --weights $G --bench math500 --samples 4 --out exp8/results/exp8_gemma-b25_$G.json); DEP=$J; echo gemma $J
if [ -n "$GS" ]; then
  T=$(python3 -c "print(round(1.0/$GS,4))")
  J=$(sub --job-name=subham_wsp_ev_gemma_ctrl --time=03:00:00 --export=ALL,EXP8_FAMILY=gemma exp8/scripts/eval.sbatch \
      --model gemma --weights base --bench math500 --samples 4 --temperature $T \
      --out exp8/results/ctrl_gemma_T$T.json); DEP=$J; echo gemma_ctrl $J
fi
# Qwen base at temperature 0.6 (the phase-2 baseline)
J=$(sub --job-name=subham_wsp_ev_base_t06 --time=24:00:00 --mem=200G exp8/scripts/eval.sbatch --weights base \
    --bench math500,gsm8k,ifeval --samples $S --out exp8/results/base_t06.json); DEP=$J; echo base_t06 $J
for pair in b50:$B50 b25:$B25 b10:$B10; do
  tag=${pair%%:*}; eid=${pair##*:}
  J=$(sub --job-name=subham_wsp_ev_$tag --time=24:00:00 --mem=200G exp8/scripts/eval.sbatch --weights $eid \
      --bench math500,gsm8k,ifeval --samples $S --out exp8/results/exp8_${tag}_$eid.json); DEP=$J; echo $tag $J
done
# matched-temperature controls (GRADING.md addition): unedited model at T/s, presence 1.5/s; MATH-500 x4
for s in 0.6 0.68; do
  T=$(python3 -c "print(round(0.6/$s,4))"); P=$(python3 -c "print(round(1.5/$s,4))")
  J=$(sub --job-name=subham_wsp_ev_ctrl_$s --time=12:00:00 --mem=200G exp8/scripts/eval.sbatch --weights base \
      --bench math500 --samples math500=4 --temperature $T --presence-penalty $P \
      --out exp8/results/ctrl_s${s}_T${T}_pp${P}.json); DEP=$J; echo ctrl_$s $J
done
squeue -u subham
