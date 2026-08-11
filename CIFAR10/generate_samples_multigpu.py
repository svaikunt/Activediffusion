"""
Multi-GPU sample generation script for Active Diffusion on CIFAR-10
Works with: torchrun --standalone --nproc_per_node=N generate_samples_multigpu.py [args]

Usage:
    torchrun --standalone --nproc_per_node=4 generate_samples_multigpu.py \
        --ckpt checkpoint.pt \
        --active \
        --num_samples 50000 \
        --batch_size 512 \
        --output_dir fid_samples
"""
import argparse
import glob
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.utils import save_image, make_grid
import inspect

from model_cifar10_sde_DDPM_v2 import (
    CIFAR10Diffusion_SDE_V2,
    CIFAR10_Active_Diffusion_SDE_V2,
)
from utils_DDPM import ExponentialMovingAverage


def _parse_int_list(csv: str):
    return [int(v.strip()) for v in csv.split(",") if v.strip()]


def _sampling_supports_pf_solver(model_to_sample) -> bool:
    sampling = getattr(model_to_sample, "sampling", None)
    if sampling is None:
        return False
    try:
        return "pf_solver" in inspect.signature(sampling).parameters
    except (TypeError, ValueError):
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU CIFAR-10 Sample Generation (v2 UNet shape)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (with model + EMA)")
    parser.add_argument("--active", action="store_true", help="Use the active diffusion model")
    parser.add_argument("--model_base_dim", type=int, default=128, help="Base dim (overridden if inferred from ckpt)")
    parser.add_argument("--dim_mults", type=str, default="1,2,2,2", help="Comma-separated UNet dim multipliers")
    parser.add_argument("--attn_resolutions", type=str, default="16,8", help="Comma-separated attention resolutions")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="UNet residual blocks per resolution (must match training)")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of SDE steps configured during training")
    parser.add_argument("--Tp", type=float, default=1e-3)
    parser.add_argument("--Ta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--num_samples", type=int, default=50000, help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=256, help="Generation batch size PER GPU")
    parser.add_argument("--output_dir", type=str, default="fid_samples", help="Where to save PNG files")
    parser.add_argument("--grid_out", type=str, default="", help="Optional path to save a grid image for visual inspection")
    parser.add_argument("--grid_count", type=int, default=100, help="Number of images in the grid (laid out as a near-square panel)")
    parser.add_argument("--grid_labels", type=str, default="", help="TXT file with 100 labels (one per line)")
    parser.add_argument("--probability_flow", action="store_true", help="Use probability-flow ODE sampler")
    parser.add_argument("--pf_steps", type=int, default=0, help="Number of ODE steps for probability-flow sampling")
    parser.add_argument(
        "--pf_schedule",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Time discretization schedule for probability-flow sampling",
    )
    parser.add_argument("--pf_solver", type=str, default="heun", choices=["heun", "rk45"], help="PF-ODE solver (heun fixed-step or rk45 adaptive)")
    parser.add_argument("--no_tweedie", action="store_true", help="Disable Tweedie denoising at final step (active and passive; not applied with --pf_solver rk45)")
    parser.add_argument("--sscs", action="store_true", help="Use the symmetric-splitting (SSCS-style) stochastic sampler instead of Euler-Maruyama (active model only, ignored with --probability_flow)")
    parser.add_argument("--score_time", type=str, default="midpoint", choices=["midpoint", "start"], help="SSCS only: score the network at the true midpoint (default, fixed) or the stale step-start time (old behavior, for A/B comparison)")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed generation if environment variables are set"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def infer_base_dim_from_state(state_dict: dict) -> Optional[int]:
    candidate_keys = [
        "model.init_conv.module.0.weight",
        "module.model.init_conv.module.0.weight",
        "model.init_conv.weight",
        "module.model.init_conv.weight",
    ]
    for key in candidate_keys:
        if key in state_dict:
            return state_dict[key].shape[0]
    return None


def detect_model_type_from_state(state_dict: dict) -> bool:
    for key in state_dict.keys():
        if "eta_scale_tensor" in key:
            return True
    candidate_keys = [
        "model.init_conv.module.0.weight",
        "module.model.init_conv.module.0.weight",
        "model.init_conv.weight",
        "module.model.init_conv.weight",
    ]
    for key in candidate_keys:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            if in_channels == 6:
                return True
            elif in_channels == 3:
                return False
    return False


def build_model(args, device, dim_mults, attn_resolutions):
    if args.active:
        model = CIFAR10_Active_Diffusion_SDE_V2(
            timesteps=args.timesteps,
            image_size=32,
            time_embedding_dim=256,
            base_dim=args.model_base_dim,
            dim_mults=dim_mults,
            attn_resolutions=attn_resolutions,
            num_res_blocks=args.num_res_blocks,
            Tp=args.Tp,
            Ta=args.Ta,
            k=args.k,
            tau=args.tau,
            T=args.T,
        )
    else:
        model = CIFAR10Diffusion_SDE_V2(
            timesteps=args.timesteps,
            image_size=32,
            in_channels=3,
            time_embedding_dim=256,
            base_dim=args.model_base_dim,
            dim_mults=dim_mults,
            attn_resolutions=attn_resolutions,
            num_res_blocks=args.num_res_blocks,
            T=args.T,
            k=args.k,
            Tp=args.Tp,
        )
    return model.to(device)


def load_ema_model(model, ckpt, device):
    model_state = ckpt["model"]
    model_keys = set(model.state_dict().keys())
    filtered_state = {}
    for ckpt_key, ckpt_value in model_state.items():
        if ckpt_key in model_keys:
            filtered_state[ckpt_key] = ckpt_value
        elif ckpt_key.startswith("module.") and ckpt_key[7:] in model_keys:
            filtered_state[ckpt_key[7:]] = ckpt_value
        elif f"module.{ckpt_key}" in model_keys:
            filtered_state[f"module.{ckpt_key}"] = ckpt_value

    model.load_state_dict(filtered_state, strict=False)
    ema = ExponentialMovingAverage(model, decay=0.0, device=device)

    ema_state = ckpt["model_ema"]
    ema_keys = set(ema.state_dict().keys())
    filtered_ema_state = {}
    for ckpt_key, ckpt_value in ema_state.items():
        if ckpt_key in ema_keys:
            filtered_ema_state[ckpt_key] = ckpt_value
        elif ckpt_key.startswith("module.") and ckpt_key[7:] in ema_keys:
            filtered_ema_state[ckpt_key[7:]] = ckpt_value
        elif f"module.{ckpt_key}" in ema_keys:
            filtered_ema_state[f"module.{ckpt_key}"] = ckpt_value
    ema.load_state_dict(filtered_ema_state, strict=False)

    ema.module.to(device)
    ema.module.eval()
    return ema.module


@torch.no_grad()
def generate_samples(model, args, rank, world_size, is_main):
    """
    Generate samples across multiple GPUs.
    Each GPU generates a portion of the total samples.
    """
    device = torch.device(f"cuda:{rank}")
    
    # Only rank 0 creates output directory
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wait for directory creation
    if world_size > 1:
        dist.barrier()
    
    # Calculate samples per GPU
    samples_per_gpu = args.num_samples // world_size
    remainder = args.num_samples % world_size
    
    # Rank 0 handles the remainder
    if rank == 0:
        my_num_samples = samples_per_gpu + remainder
    else:
        my_num_samples = samples_per_gpu
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"Multi-GPU Sample Generation")
        print(f"{'='*60}")
        print(f"Total samples: {args.num_samples}")
        print(f"GPUs: {world_size}")
        print(f"Samples per GPU: {samples_per_gpu} (rank 0: {samples_per_gpu + remainder})")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Probability flow: {args.probability_flow}")
        if args.probability_flow:
            print(f"PF steps: {args.pf_steps if args.pf_steps > 0 else 'default'}")
            print(f"PF schedule: {args.pf_schedule}")
            print(f"PF solver: {args.pf_solver}")
        elif args.active:
            print(f"Stochastic sampler: {'SSCS (exact-linear split)' if args.sscs else 'Euler-Maruyama'}")
        print(f"{'='*60}\n")
    
    # Generate samples
    total = 0
    sample_idx = rank * samples_per_gpu  # Each GPU has its own index range
    
    # pf_steps/pf_schedule now control the step count/schedule for every sampler
    # (SDE and SSCS included, not just PF-ODE) -- see model_cifar10_sde_DDPM.py.
    pf_steps_val = args.pf_steps if args.pf_steps > 0 else None
    pf_kwargs = {"pf_steps": pf_steps_val, "pf_schedule": args.pf_schedule}
    if args.probability_flow:
        pf_kwargs["pf_solver"] = args.pf_solver
        if not _sampling_supports_pf_solver(model):
            pf_kwargs.pop("pf_solver", None)

    while total < my_num_samples:
        current_batch = min(args.batch_size, my_num_samples - total)

        if args.active and args.sscs and not args.probability_flow:
            images, _ = model.sampling_sscs(
                current_batch,
                device=device,
                tweedie=not args.no_tweedie,
                pf_steps=pf_steps_val,
                pf_schedule=args.pf_schedule,
                score_time=args.score_time,
            )
        elif args.active:
            images, _ = model.sampling(
                current_batch,
                device=device,
                probability_flow=args.probability_flow,
                tweedie=not args.no_tweedie,
                **pf_kwargs,
            )
        else:
            images = model.sampling(
                current_batch,
                device=device,
                probability_flow=args.probability_flow,
                tweedie=not args.no_tweedie,
                **pf_kwargs,
            )

        # Each GPU saves its own images
        for i in range(current_batch):
            save_image(images[i], os.path.join(args.output_dir, f"{sample_idx:05d}.png"), normalize=False)
            sample_idx += 1

        total += current_batch
        
        if is_main:
            print(f"[Rank {rank}] Saved {total}/{my_num_samples} samples", flush=True)
    
    # Synchronize all GPUs
    if world_size > 1:
        dist.barrier()
    
    if is_main:
        print(f"\n{'='*60}")
        print(f"All {args.num_samples} samples saved to {args.output_dir}")
        print(f"{'='*60}")
    
    # Optional: Create grid (only rank 0)
    if is_main and args.grid_out:
        print("\nCreating sample grid...")
        image_paths = sorted(glob.glob(os.path.join(args.output_dir, "*.png")))
        if len(image_paths) < args.grid_count:
            print(f"Warning: Need at least {args.grid_count} samples for grid, found {len(image_paths)}")
        else:
            selected = image_paths[:args.grid_count]
            tensors = []
            for path in selected:
                img = Image.open(path).convert("RGB")
                tensors.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
            grid_images = torch.stack(tensors, dim=0)
            nrow = math.ceil(math.sqrt(args.grid_count))
            grid = make_grid(grid_images, nrow=nrow, normalize=False)
            save_image(grid, args.grid_out)
            print(f"Grid of {args.grid_count} images saved to {args.grid_out}")


def main():
    args = parse_args()
    
    # Setup distributed
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        if is_main:
            print("ERROR: CUDA not available. This script requires GPU.")
        return
    
    # Load checkpoint (all ranks load to avoid I/O bottleneck)
    if is_main:
        print(f"Loading checkpoint: {args.ckpt}")
    
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    # Auto-detect model type and base_dim
    if not args.active:
        detected_active = detect_model_type_from_state(ckpt.get("model", {}))
        if detected_active:
            if is_main:
                print("Detected active model from checkpoint. Using active diffusion model.")
            args.active = True
    
    inferred_dim = infer_base_dim_from_state(ckpt.get("model", {}))
    if inferred_dim is not None and inferred_dim != args.model_base_dim:
        if is_main:
            print(f"Inferred base_dim {inferred_dim} from checkpoint. Overriding CLI value {args.model_base_dim}.")
        args.model_base_dim = inferred_dim
    
    dim_mults = _parse_int_list(args.dim_mults)
    attn_resolutions = tuple(_parse_int_list(args.attn_resolutions))
    
    # Build model
    model = build_model(args, device, dim_mults, attn_resolutions)
    ema_model = load_ema_model(model, ckpt, device)
    
    if is_main:
        num_params = sum(p.numel() for p in ema_model.parameters())
        print(f"Model loaded successfully!")
        print(f"Parameters: {num_params:,}")
        print(f"Model type: {'Active' if args.active else 'Passive'}")
    
    # Generate samples
    generate_samples(ema_model, args, rank, world_size, is_main)
    
    if is_distributed:
        dist.destroy_process_group()
    
    if is_main:
        print("\nDone!")


if __name__ == "__main__":
    main()

