#!/bin/bash
#SBATCH --job-name=t05cond
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=t05cond_%j.log
set -u
CKPT=${1:-/net/scratch/svaikunt/cifar10/active_cld_tau05_cond/checkpoint_epoch_800.pt}
[ -f "$CKPT" ] || { echo "FATAL: $CKPT not found" >&2; exit 1; }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10
export TMPDIR=/net/scratch/svaikunt/tmp
mkdir -p "$TMPDIR"
echo "resuming from $CKPT"
python -u train_multigpu.py \
  --active --score_param cond --amp \
  --ckpt "$CKPT" \
  --out_dir /net/scratch/svaikunt/cifar10/active_cld_tau05_cond \
  --Tp 1e-3 --Ta 6.4 --tau 0.5 --k 4.0 --T 2.0 \
  --epochs 2000 --batch_size 128 \
  --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
  --large_sample_interval 200 --large_sample_count 200 \
  --pf_sample_interval 200 --pf_sample_count 200 --pf_steps 400 --pf_schedule log \
  --model_ema_decay 0.9984375 --model_ema_steps 10 \
  --save_freq 100 \
  --lr 0.0002 --warmup_steps 100000 --timesteps 1000 \
  --grad_clip 1.0 --weight_decay 0.0 \
  --data_dir /home/svaikunt/CIFAR10/data
