"""
Save the MNIST test set (10 000 images) as individual PNG files so that
pytorch-fid can compute Inception statistics over the real distribution.

Usage:
    python save_real_mnist.py --output_dir fid_data/real
"""

import argparse
import os
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="fid_data/real",
                        help="Directory to save real MNIST PNGs")
    parser.add_argument("--data_root", type=str, default="./mnist_data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = MNIST(root=args.data_root, train=False, download=True,
                    transform=transforms.ToTensor())

    for i, (img, _) in enumerate(dataset):
        arr = (img.squeeze().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(
            os.path.join(args.output_dir, f"{i:05d}.png")
        )
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(dataset)}")

    print(f"Saved {len(dataset)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
