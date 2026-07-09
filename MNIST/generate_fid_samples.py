"""
Load a diffusion model checkpoint and save generated samples as PNG files,
ready for FID evaluation with pytorch-fid.

Usage:
    python generate_fid_samples.py \\
        --ckpt results/Passive_60_T1.0_k1.0.pt \\
        --output_dir fid_data/passive_60 \\
        --n_samples 10000
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image

from FID_comparison import load_ema_from_checkpoint, generate_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated PNGs")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of images to generate (default: 10000)")
    parser.add_argument("--sample_batch_size", type=int, default=256,
                        help="Batch size for generation (tune to GPU memory)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    model_ema, model_type = load_ema_from_checkpoint(args.ckpt, device)
    print(f"Model type: {model_type}")

    print(f"Generating {args.n_samples} samples...")
    samples = generate_samples(
        model_ema.module, model_type, args.n_samples, device, args.sample_batch_size
    )
    # samples: float tensor [N, 1, 28, 28] in [0, 1]

    print(f"Saving PNGs to {args.output_dir} ...")
    for i, img in enumerate(samples):
        arr = (img.squeeze().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(
            os.path.join(args.output_dir, f"{i:05d}.png")
        )
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{args.n_samples}")

    print(f"Done. {args.n_samples} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
