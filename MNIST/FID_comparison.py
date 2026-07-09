"""
FID_comparison.py

Compute FID scores for passive, active, and linear diffusion models at every
saved checkpoint epoch, then plot FID vs epoch on a single figure.

Features extracted from the penultimate (84-dim f6) layer of a pretrained
LeNet5 classifier, which serves as a perceptually-meaningful feature space
for MNIST in place of the Inception v3 used in standard FID.

Usage example:
    python FID_comparison.py \\
        --passive "results/Passive_*_T1.0_k1.0.pt" \\
        --active  "results/Active_*_Tp0.001_Ta1.0_tau0.5.pt" \\
        --linear  "results/Linear_*_fixed_M.pt" \\
        --lenet_weights weights/lenet_epoch=12_test_acc=0.991.pth \\
        --n_samples 5000 \\
        --output fid_comparison.png
"""

import argparse
import glob
import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from lenet import LeNet5
from model import MNISTDiffusion, MNIST_Active_Diffusion, MNIST_Linear_Diffusion
from utils import ExponentialMovingAverage


# ── LeNet5 feature extractor ──────────────────────────────────────────────────

class LeNetFeatureExtractor(nn.Module):
    """
    Wraps a pretrained LeNet5 and returns 84-dim f6 activations.
    LeNet5 expects 1x32x32 input, so 28x28 images are zero-padded by 2 pixels
    on each side before being passed through the network.
    """
    def __init__(self, lenet):
        super().__init__()
        self.convnet = lenet.convnet
        self.f6 = lenet.fc.f6
        self.relu6 = lenet.fc.relu6

    def forward(self, x):
        x = nn.functional.pad(x, (2, 2, 2, 2))  # 28x28 → 32x32
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        return self.relu6(self.f6(x))             # [B, 84]


def load_feature_extractor(weights_path, device):
    lenet = LeNet5()
    lenet.load_state_dict(torch.load(weights_path, map_location=device))
    lenet.eval()
    extractor = LeNetFeatureExtractor(lenet).to(device)
    extractor.eval()
    return extractor


# ── FID helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(extractor, images, device, batch_size=512):
    """
    Pass images through the feature extractor in batches.
    images: float tensor [N, 1, 28, 28] in [-1, 1].
    Returns: numpy array [N, 84].
    """
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        feats.append(extractor(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def real_mnist_features(extractor, device, data_root="./mnist_data", batch_size=512):
    """Compute LeNet5 features for the full MNIST test set (10 000 images)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MNIST(root=data_root, train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            feats.append(extractor(imgs.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu_r, sigma_r, feats_gen):
    """
    FID = ||mu_r - mu_g||^2 + Tr(sigma_r + sigma_g - 2 * sqrt(sigma_r @ sigma_g))
    mu_r, sigma_r: pre-computed statistics of the real distribution.
    feats_gen: numpy array [N, D] of generated-image features.
    """
    mu_g = feats_gen.mean(0)
    sigma_g = np.cov(feats_gen, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))


# ── Model construction from saved args ────────────────────────────────────────

def build_model(saved_args, device):
    """Re-instantiate a model from the args dict stored in a checkpoint."""
    mt  = saved_args["model_type"]
    bs  = saved_args["model_base_dim"]
    ts  = saved_args["timesteps"]
    tt  = saved_args["total_time"]

    if mt == "passive":
        return MNISTDiffusion(
            timesteps=ts, image_size=28, in_channels=1,
            base_dim=bs, dim_mults=[2, 4],
            T=saved_args["T"],
            k=saved_args.get("k", 1.0),
            total_time=tt,
        ).to(device)

    if mt == "active":
        return MNIST_Active_Diffusion(
            timesteps=ts, image_size=28, time_embedding_dim=256,
            base_dim=bs, dim_mults=[2, 4],
            Tp=saved_args["Tp"], Ta=saved_args["Ta"],
            k=saved_args["k"], tau=saved_args["tau"],
            total_time=tt,
        ).to(device)

    if mt == "linear":
        return MNIST_Linear_Diffusion(
            image_size=28,
            M_init=saved_args["M"], D=saved_args["D"],
            learn_M=saved_args.get("learn_M", False),
            M_constraint=saved_args.get("M_constraint", "none"),
            M_norm_radius=saved_args.get("M_norm_radius"),
            M_max_norm=saved_args.get("M_max_norm"),
            stability_weight=saved_args.get("stability_weight", 10.0),
            M_l2_weight=saved_args.get("M_l2_weight", 0.01),
            eta0_mode=saved_args.get("eta0_mode", "stationary_marginal"),
            eta0_value=saved_args.get("eta0_value", 0.0),
            time_embedding_dim=256, timesteps=ts,
            base_dim=bs, dim_mults=[2, 4], total_time=tt,
        ).to(device)

    raise ValueError(f"Unknown model_type: {mt!r}")


def load_ema_from_checkpoint(ckpt_path, device):
    """Load the EMA model from a checkpoint. Returns (model_ema, model_type)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]
    model = build_model(saved_args, device)
    # Replicate the EMA decay used at training time
    adjust = saved_args["batch_size"] * saved_args["model_ema_steps"] / saved_args["epochs"]
    alpha = min(1.0, (1.0 - saved_args["model_ema_decay"]) * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)
    model_ema.load_state_dict(ckpt["model_ema"])
    model_ema.eval()
    return model_ema, saved_args["model_type"]


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoints(pattern):
    """
    Glob for files matching pattern and sort by epoch number.
    The epoch is the first integer after the model-type prefix, e.g.:
        Passive_30_T1.0_k1.0.pt  →  epoch 30
    Returns a sorted list of (epoch, path) tuples.
    Skips files whose name does not contain a recognisable epoch number.
    """
    found = []
    for path in glob.glob(pattern):
        m = re.search(r'_(\d+)_', os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    found.sort()
    return found


# ── Sample generation ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(module, model_type, n_samples, device, batch_size=256):
    """
    Generate n_samples images from module.sampling() in batches to avoid OOM.
    Returns a CPU tensor [N, 1, 28, 28] in [0, 1].
    """
    collected = []
    remaining = n_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        if model_type == "passive":
            samples = module.sampling(n, device=device)
        else:
            samples, _ = module.sampling(n, device=device)
        collected.append(samples.cpu())
        remaining -= n
    return torch.cat(collected, dim=0)


# ── FID at a single checkpoint ────────────────────────────────────────────────

def compute_fid_for_checkpoint(ckpt_path, extractor, mu_r, sigma_r,
                                n_samples, sample_batch_size, device):
    model_ema, model_type = load_ema_from_checkpoint(ckpt_path, device)
    samples = generate_samples(
        model_ema.module, model_type, n_samples, device, sample_batch_size
    )
    # Samples are in [0, 1]; rescale to [-1, 1] to match LeNet's normalisation
    samples = samples * 2.0 - 1.0
    gen_feats = extract_features(extractor, samples, device)
    return frechet_distance(mu_r, sigma_r, gen_feats)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FID vs epoch plot for passive / active / linear diffusion"
    )
    p.add_argument("--passive", type=str, default=None,
                   help='Glob pattern for passive checkpoints, '
                        'e.g. "results/Passive_*_T1.0_k1.0.pt"')
    p.add_argument("--active",  type=str, default=None,
                   help='Glob pattern for active checkpoints, '
                        'e.g. "results/Active_*_Tp0.001_Ta1.0_tau0.5.pt"')
    p.add_argument("--linear",  type=str, default=None,
                   help='Glob pattern for linear checkpoints, '
                        'e.g. "results/Linear_*_fixed_M.pt"')
    p.add_argument("--lenet_weights", type=str,
                   default="weights/lenet_epoch=12_test_acc=0.991.pth",
                   help="Path to pretrained LeNet5 weights")
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Total generated samples per checkpoint (default: 5000)")
    p.add_argument("--sample_batch_size", type=int, default=256,
                   help="Batch size for sample generation (tune to fit GPU memory)")
    p.add_argument("--data_root", type=str, default="./mnist_data",
                   help="Root directory for MNIST data")
    p.add_argument("--output", type=str, default="fid_comparison.png",
                   help="Output plot filename")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    # Feature extractor
    print("Loading LeNet5 feature extractor...")
    extractor = load_feature_extractor(args.lenet_weights, device)

    # Real MNIST statistics (computed once, reused for every checkpoint)
    print("Extracting real MNIST test-set features...")
    real_feats = real_mnist_features(extractor, device, data_root=args.data_root)
    mu_r    = real_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    print(f"  Real feature matrix: {real_feats.shape}  "
          f"(mean={mu_r.mean():.4f}, std={real_feats.std():.4f})")

    # Discover and evaluate checkpoints for each model type
    model_configs = [
        ("Passive", args.passive),
        ("Active",  args.active),
        ("Linear",  args.linear),
    ]
    colors = {"Passive": "steelblue", "Active": "tomato", "Linear": "seagreen"}

    results = {}
    for label, pattern in model_configs:
        if pattern is None:
            print(f"\n[{label}] No pattern provided — skipping.")
            continue

        checkpoints = find_checkpoints(pattern)
        if not checkpoints:
            print(f"\n[{label}] No checkpoints matched: {pattern}")
            continue

        print(f"\n[{label}] {len(checkpoints)} checkpoint(s) found.")
        epochs, fids = [], []
        for epoch, ckpt_path in checkpoints:
            print(f"  Epoch {epoch:4d}  {os.path.basename(ckpt_path)} ... ",
                  end="", flush=True)
            fid = compute_fid_for_checkpoint(
                ckpt_path, extractor, mu_r, sigma_r,
                args.n_samples, args.sample_batch_size, device,
            )
            print(f"FID = {fid:.2f}")
            epochs.append(epoch)
            fids.append(fid)

        results[label] = (epochs, fids)

    if not results:
        print("\nNo results to plot — check your glob patterns.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (epochs, fids) in results.items():
        ax.plot(epochs, fids, marker="o", linewidth=1.8,
                label=label, color=colors[label])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("FID", fontsize=12)
    ax.set_title("FID vs Epoch — Passive / Active / Linear Diffusion", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
