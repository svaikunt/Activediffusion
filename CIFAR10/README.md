# CIFAR-10 Diffusion — Passive & Active

Score-based generative models on CIFAR-10 using two SDE formulations:

- **Passive** — Ornstein–Uhlenbeck process: `dx = -k·x dt + √(2Tp) dW`
- **Active** — Active-matter SDE driven by a coloured-noise field η:
  ```
  dx = (-k·x + η) dt + √(2Tp) dW_x
  dη = (-η/τ)    dt + (1/τ)√(2Ta) dW_η
  ```

Both models share the same UNet score network ([unet_DDPM.py](unet_DDPM.py)) and training loop ([train_multigpu.py](train_multigpu.py)).

---

## Files

| File | Purpose |
|---|---|
| `train_multigpu.py` | Multi-GPU training script (`torchrun`) |
| `generate_samples_multigpu.py` | Multi-GPU sample generation + optional PF-ODE sampler |
| `model_cifar10_sde_DDPM.py` | Base passive and active diffusion model classes |
| `model_cifar10_sde_DDPM_v2.py` | V2 wrappers with configurable `attn_resolutions` |
| `unet_DDPM.py` | UNet score network |
| `utils_DDPM.py` | EMA utility |
| `run_experiments.ipynb` | End-to-end notebook: train → generate → FID |

---

## Quickstart

The easiest way to run everything is the notebook:

```bash
jupyter notebook run_experiments.ipynb
```

It trains passive and active models in two phases each, generates 40 000 samples, and computes FID — all in sequential cells.

---

## Training

Training is split into two phases for each model to anneal the learning rate and tighten the EMA.

### Passive — Phase 1 (epochs 1–1000)

```bash
torchrun --standalone --nproc_per_node=4 train_multigpu.py \
  --out_dir results_cifar10_passive \
  --epochs 1000 --batch_size 128 \
  --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
  --lr 0.0001 --Tp 6.4 --k 4.0 --T 2.0 \
  --model_ema_decay 0.997 --save_freq 200 \
  --warmup_steps 5000 --timesteps 1000 --grad_clip 1.0 --amp
```

### Passive — Phase 2 (epochs 1001–2600)

```bash
torchrun --standalone --nproc_per_node=4 train_multigpu.py \
  --ckpt results_cifar10_passive/checkpoint_epoch_1000.pt \
  --out_dir results_cifar10_passive \
  --epochs 2600 --batch_size 128 \
  --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
  --lr 0.00005 --Tp 6.4 --k 4.0 --T 2.0 \
  --model_ema_decay 0.9997 --save_freq 200 \
  --warmup_steps 5000 --timesteps 1000 --grad_clip 1.0 --amp
```

### Active — Phase 1 (epochs 1–1000)

```bash
torchrun --standalone --nproc_per_node=4 train_multigpu.py \
  --active --amp \
  --out_dir results_cifar10_active \
  --epochs 1000 --batch_size 128 \
  --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
  --lr 0.0001 --Tp 1e-3 --Ta 6.4 --k 4.0 --tau 0.15 --T 2.0 \
  --model_ema_decay 0.997 --save_freq 200 \
  --warmup_steps 5000 --timesteps 1000 --grad_clip 1.0 --weight_decay 0.0
```

### Active — Phase 2 (epochs 1001–2600)

```bash
torchrun --standalone --nproc_per_node=4 train_multigpu.py \
  --active --amp \
  --ckpt results_cifar10_active/checkpoint_epoch_1000.pt \
  --out_dir results_cifar10_active \
  --epochs 2600 --batch_size 128 \
  --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
  --lr 0.00005 --Tp 1e-3 --Ta 6.4 --k 4.0 --tau 0.15 --T 2.0 \
  --model_ema_decay 0.9997 --save_freq 200 \
  --warmup_steps 5000 --timesteps 1000 --grad_clip 1.0 --weight_decay 0.0
```

> Adjust `--nproc_per_node` to the number of GPUs available (`nvidia-smi`).  
> Checkpoints are saved every `--save_freq` epochs as `checkpoint_epoch_NNN.pt` inside `--out_dir`.

---

## Sample Generation

```bash
# Passive (probability-flow ODE)
torchrun --standalone --nproc_per_node=4 generate_samples_multigpu.py \
  --ckpt results_cifar10_passive/checkpoint_epoch_2600.pt \
  --model_base_dim 128 --num_res_blocks 4 --attn_resolutions 16 \
  --num_samples 40000 --batch_size 512 \
  --output_dir results_cifar10_passive/fid_samples_2600 \
  --Tp 6.4 --k 4.0 --T 2.0 \
  --probability_flow --pf_steps 600 --pf_schedule log

# Active (SDE sampler)
torchrun --standalone --nproc_per_node=4 generate_samples_multigpu.py \
  --ckpt results_cifar10_active/checkpoint_epoch_2600.pt \
  --active \
  --model_base_dim 128 --num_res_blocks 4 --attn_resolutions 16 \
  --num_samples 40000 --batch_size 512 \
  --output_dir results_cifar10_active/fid_samples_2600 \
  --Tp 1e-3 --Ta 6.4 --k 4.0 --tau 0.15
```

---

## FID

```python
from cleanfid import fid

score = fid.compute_fid(
    fdir1="results_cifar10_passive/fid_samples_2600",
    dataset_name="cifar10",
    dataset_res=32,
    dataset_split="train",
    mode="legacy_pytorch",
    model_name="inception_v3",
)
```

Install: `pip install clean-fid`