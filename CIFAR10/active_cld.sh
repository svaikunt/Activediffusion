#!/bin/bash
#SBATCH --job-name=active_cld
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=active_cld_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10

# Single GPU, CLD-paper optimization (ICLR 2022, Table 6, CIFAR-10 Main):
#   lr 2e-4 Adam, EMA 0.9999, 100k warmup, grad-clip 1.0, batch 128, AMP.
# batch 128 on one GPU -> 390 steps/epoch; 2000 epochs = 781k steps ~ CLD's 800k.
#
# --model_ema_decay 0.9984375 is the NOMINAL value that yields CLD's effective
# 0.9999 through the script's torchvision adjustment:
#   adjust = 128*10/2000 = 0.64 ; alpha = (1-0.9984375)*0.64 = 1e-3
#   -> 10k-step horizon, i.e. per-iteration decay 0.9999.
# It depends on --epochs, --batch_size and --model_ema_steps: re-derive if any change.
ARGS=(
  --active
  --amp
  --out_dir active_cld_lr2e4
  --Tp 1e-3
  --Ta 6.4
  --tau 0.15
  --k 4.0
  --T 2.0
  --epochs 2000
  --batch_size 128
  --model_base_dim 128
  --num_res_blocks 4
  --dim_mults 1,2,2,2
  --attn_resolutions 16
  --large_sample_interval 500
  --large_sample_count 200
  --pf_sample_interval 500
  --pf_sample_count 200
  --pf_steps 500
  --pf_schedule log
  --model_ema_decay 0.9984375
  --model_ema_steps 10
  --save_freq 500
  --lr 0.0002
  --warmup_steps 100000
  --timesteps 1000
  --grad_clip 1.0
  --weight_decay 0.0
  --data_dir /home/svaikunt/CIFAR10/data
)

echo "ARGS: ${ARGS[@]}"
python -u train_multigpu.py "${ARGS[@]}"
