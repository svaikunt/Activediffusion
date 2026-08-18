#!/bin/bash
#SBATCH --job-name=fidab4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=fidab4_%j.log
set -u
EPOCH=${1:-1000}
N=${2:-50000}
PF=${3:-500}
NG=4
PER=$(( N / NG ))
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
for ARM in cond score; do
  case $ARM in cond) CK=$COND ;; score) CK=$SCORE ;; esac
  echo "=== $ARM  epoch $EPOCH  N=$N  PF-$PF  ${NG} GPUs x ${PER} ==="
  rm -rf fid${B}_${ARM} fid${B}_${ARM}_r0 fid${B}_${ARM}_r1 fid${B}_${ARM}_r2 fid${B}_${ARM}_r3
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u generate_samples_multigpu.py \
      --ckpt "$CK" $SDE --score_param "$ARM" $ARCH $SAMP \
      --num_samples $PER --batch_size 256 \
      --output_dir "fid${B}_${ARM}_r${i}" \
      > "log_${B}_${ARM}_r${i}.txt" 2>&1 &
  done
  wait
  python - "$B" "$ARM" "$NG" <<'PYEOF'
import glob, os, sys
b, arm, ng = sys.argv[1], sys.argv[2], int(sys.argv[3])
dst = f"fid{b}_{arm}"; os.makedirs(dst, exist_ok=True)
n = 0
for i in range(ng):
    for f in sorted(glob.glob(f"{dst}_r{i}/*.png")):
        os.rename(f, os.path.join(dst, f"{n:06d}.png")); n += 1
print(f"merged {n} into {dst}", flush=True)
PYEOF
  T=$(find "fid${B}_${ARM}" -name '*.png' | wc -l)
  echo "$ARM total = $T"
  [ "$T" -lt "$N" ] && { echo "INCOMPLETE $ARM -- see log_${B}_${ARM}_r*.txt" >&2; exit 1; }
  rm -rf fid${B}_${ARM}_r0 fid${B}_${ARM}_r1 fid${B}_${ARM}_r2 fid${B}_${ARM}_r3
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
