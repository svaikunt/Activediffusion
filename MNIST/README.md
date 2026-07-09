# MNIST Diffusion — Passive, Active & Linear

Score-based generative models on MNIST using three SDE formulations:

- **Passive** — Ornstein–Uhlenbeck process: `dx = -k x dt + √(2T) dW`
- **Active** — Active-matter SDE driven by a coloured-noise field η:
  ```
  dx = (-k x + η) dt + √(2Tp) dW_x
  dη = (-η/τ)    dt + (1/τ)√(2Ta) dW_η
  ```
- **Linear** — General linear SDE `dz = M z dt + D dW`, `z = (x, η)`, with optional learnable drift matrix M

All three share the same UNet score network ([unet.py](unet.py)) and training loop ([train_mnist_fixed_normM.py](train_mnist_fixed_normM.py)).

---

## Files

| File | Purpose |
|---|---|
| `train_mnist_fixed_normM.py` | Training script for all three model types |
| `generate_fid_samples.py` | Generate PNG samples from a checkpoint for FID evaluation |
| `save_real_mnist.py` | Save MNIST test-set images as PNGs (run once before FID) |
| `model.py` | Passive, active, and linear diffusion model classes |
| `unet.py` | UNet score network |
| `utils.py` | EMA utility |
| `FID_comparison.py` | Checkpoint loading + sampling utilities (used by `generate_fid_samples.py`) |
| `lenet.py` | LeNet5 classifier (imported by `FID_comparison.py`; not invoked in the pytorch-fid path) |
| `run_experiments.ipynb` | End-to-end notebook: train → generate → FID |
| `train_mnist.py` | Original single-model training script (superseded by `train_mnist_fixed_normM.py`) |

---

## Quickstart

The easiest way to run everything is the notebook:

```bash
jupyter notebook run_experiments.ipynb
```

It trains all three models (optionally in parallel across GPUs), generates samples, and computes pytorch-fid scores — including a multi-checkpoint FID-vs-epoch sweep with a table and plot.

---

## Training

Each model trains for 60 epochs. Checkpoints are saved every 5 epochs to `results/` as `<ModelType>_<epoch>_<params>.pt`.

Pin each job to a dedicated GPU with `CUDA_VISIBLE_DEVICES=N`.

### Passive (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 python train_mnist_fixed_normM.py \
    --model_type passive \
    --T 1.0 --k 1.0 --total_time 2.0 \
    --epochs 60 --batch_size 128 --lr 0.001 \
    --timesteps 1000 --model_base_dim 64 --save_freq 5 --n_samples 36
```

### Active (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 python train_mnist_fixed_normM.py \
    --model_type active \
    --Tp 0.001 --Ta 1.0 --tau 0.5 --k 1.0 --total_time 2.0 \
    --epochs 60 --batch_size 128 --lr 0.001 \
    --timesteps 1000 --model_base_dim 64 --save_freq 5 --n_samples 36
```

### Linear — learnable M, fixed Frobenius norm (GPU 2)

M and D are initialised to the active-matter special case (`k=1`, `τ=0.5`, `Ta=1`).  
`--M_constraint fixed_norm` keeps `‖M‖_F` constant and learns only the direction.

```bash
CUDA_VISIBLE_DEVICES=2 python train_mnist_fixed_normM.py \
    --model_type linear \
    --M -1.0 1.0 0.0 -2.0 --D 0.0 0.0 0.0 2.828 \
    --learn_M --M_constraint fixed_norm \
    --M_lr_factor 0.01 --stability_weight 10.0 --M_l2_weight 0.01 \
    --eta0_mode stationary_marginal --total_time 2.0 \
    --epochs 60 --batch_size 128 --lr 0.001 \
    --timesteps 1000 --model_base_dim 64 --save_freq 5 --n_samples 36 \
    --name_suffix learnM_fixed_norm
```

To resume from a checkpoint, add `--restart yes --restart_epoch <N>`.

---

## Sample Generation

Samples are saved as individual PNGs ready for FID evaluation.

```bash
# Passive
CUDA_VISIBLE_DEVICES=0 python generate_fid_samples.py \
    --ckpt results/Passive_60_T1.0_k1.0.pt \
    --output_dir fid_data/passive_60 \
    --n_samples 10000 --sample_batch_size 256

# Active
CUDA_VISIBLE_DEVICES=0 python generate_fid_samples.py \
    --ckpt results/Active_60_Tp0.001_Ta1.0_tau0.5.pt \
    --output_dir fid_data/active_60 \
    --n_samples 10000 --sample_batch_size 256

# Linear (learn M, fixed norm)
CUDA_VISIBLE_DEVICES=0 python generate_fid_samples.py \
    --ckpt results/Linear_60_learnM_fixed_norm.pt \
    --output_dir fid_data/linear_learnM_60 \
    --n_samples 10000 --sample_batch_size 256
```

---

## FID

FID is computed with [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (Inception v3 features) against the MNIST test set.

### Step 1 — Save real MNIST test images *(once)*

```bash
python save_real_mnist.py --output_dir fid_data/real
```

### Step 2 — Precompute real Inception statistics *(once)*

```bash
python -m pytorch_fid --save-stats fid_data/real fid_data/real_stats.npz --device cuda:0
```

### Step 3 — Compute FID

```bash
python -m pytorch_fid fid_data/real_stats.npz fid_data/passive_60  --device cuda:0
python -m pytorch_fid fid_data/real_stats.npz fid_data/active_60   --device cuda:0
python -m pytorch_fid fid_data/real_stats.npz fid_data/linear_learnM_60 --device cuda:0
```

Install: `pip install pytorch-fid`

> **Note:** Inception v3 upsamples 28×28 grayscale to 299×299 RGB. FID values are valid for cross-model comparison but differ from a MNIST-native feature space.
