#!/bin/bash
#SBATCH --job-name=fidab
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=fidab_%j.log
set -u
EPOCH=${1:-800}
N=${2:-10000}
PF=${3:-300}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10
export TMPDIR=/net/scratch/svaikunt/tmp
mkdir -p "$TMPDIR"
COND=/net/scratch/svaikunt/cifar10/active_cld_tau05_cond/checkpoint_epoch_${EPOCH}.pt
SCORE=/net/scratch/svaikunt/cifar10/active_cld_tau05/checkpoint_epoch_${EPOCH}.pt
for C in "$COND" "$SCORE"; do
  [ -f "$C" ] || { echo "FATAL: $C not found" >&2; exit 1; }
done
SDE="--active --Tp 1e-3 --Ta 6.4 --tau 0.5 --k 4.0 --T 2.0"
ARCH="--model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16"
SAMP="--probability_flow --pf_steps $PF --pf_schedule quadratic"
B=${EPOCH}_pf${PF}_${N}
rm -rf fid${B}_cond fid${B}_score
for ARM in cond score; do
  case $ARM in cond) CK=$COND ;; score) CK=$SCORE ;; esac
  echo "=== $ARM  epoch $EPOCH  N=$N  PF-$PF  ==="
  CUDA_VISIBLE_DEVICES=0 python -u generate_samples_multigpu.py \
    --ckpt "$CK" $SDE --score_param "$ARM" $ARCH $SAMP \
    --num_samples $N --batch_size 256 --output_dir "fid${B}_${ARM}" \
    --grid_out grid_${B}_${ARM}.png --grid_count 400
  T=$(find "fid${B}_${ARM}" -name '*.png' | wc -l)
  echo "$ARM total = $T"
  [ "$T" -lt "$N" ] && { echo "INCOMPLETE $ARM" >&2; exit 1; }
done
python -u - "$B" <<'PYEOF'
import sys
from cleanfid import fid
b = sys.argv[1]; out = {}
for t in ("cond", "score"):
    out[t] = fid.compute_fid(fdir1=f"fid{b}_{t}", dataset_name="cifar10",
        dataset_res=32, dataset_split="train", mode="legacy_pytorch",
        model_name="inception_v3", num_workers=0)
    print(f"RESULT  tau05_{t}_{b}_quad  FID = {out[t]:.2f}", flush=True)
print(f"RESULT  delta (cond-score) = {out['cond']-out['score']:+.2f}", flush=True)
PYEOF
