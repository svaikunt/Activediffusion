import argparse
import glob
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image, make_grid

from model_cifar10_sde_DDPM_v2 import (
    CIFAR10Diffusion_SDE_V2,
    CIFAR10_Active_Diffusion_SDE_V2,
)
from utils_DDPM import ExponentialMovingAverage


def _parse_int_list(csv: str):
    return [int(v.strip()) for v in csv.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 samples for FID evaluation (v2 UNet shape)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (with model + EMA)")
    parser.add_argument("--active", action="store_true", help="Use the active diffusion model")
    parser.add_argument("--model_base_dim", type=int, default=128, help="Base dim (overridden if inferred from ckpt)")
    parser.add_argument("--dim_mults", type=str, default="1,2,2,2", help="Comma-separated UNet dim multipliers")
    parser.add_argument("--attn_resolutions", type=str, default="16,8", help="Comma-separated attention resolutions")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of SDE steps configured during training")
    parser.add_argument("--Tp", type=float, default=1e-3)
    parser.add_argument("--Ta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--num_samples", type=int, default=50000, help="How many images to generate")
    parser.add_argument("--batch_size", type=int, default=256, help="Generation batch size")
    parser.add_argument("--output_dir", type=str, default="fid_samples_v2", help="Where to save PNG files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--grid_out", type=str, default="", help="Optional path to save a 10x10 grid")
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
    return parser.parse_args()


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


def collect_grid_samples(output_dir):
    image_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    if len(image_paths) < 100:
        raise ValueError("Need at least 100 samples to build a 10x10 grid.")
    selected = image_paths[:100]
    tensors = []
    for path in selected:
        img = Image.open(path).convert("RGB")
        tensors.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def generate_samples(model, args):
    os.makedirs(args.output_dir, exist_ok=True)
    total = 0
    existing_files = sorted(glob.glob(os.path.join(args.output_dir, "*.png")))
    sample_idx = len(existing_files)
    if sample_idx > 0:
        print(f"Found {sample_idx} existing samples in {args.output_dir}; appending new images.")

    while total < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - total)
        pf_kwargs = {}
        if args.probability_flow:
            pf_kwargs = {
                "pf_steps": args.pf_steps if args.pf_steps > 0 else None,
                "pf_schedule": args.pf_schedule,
                "pf_solver": args.pf_solver,
            }

        if args.active:
            images, _ = model.sampling(
                current_batch,
                device=args.device,
                probability_flow=args.probability_flow,
                **pf_kwargs,
            )
        else:
            images = model.sampling(
                current_batch,
                device=args.device,
                probability_flow=args.probability_flow,
                **pf_kwargs,
            )

        for i in range(current_batch):
            save_image(images[i], os.path.join(args.output_dir, f"{sample_idx:05d}.png"), normalize=False)
            sample_idx += 1

        total += current_batch
        print(f"Saved {total}/{args.num_samples} samples", flush=True)

    if args.grid_out:
        grid_images = collect_grid_samples(args.output_dir)
        grid = make_grid(grid_images, nrow=10, normalize=False)
        save_image(grid, args.grid_out)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or mps.")

    ckpt = torch.load(args.ckpt, map_location="cpu")

    if not args.active:
        detected_active = detect_model_type_from_state(ckpt.get("model", {}))
        if detected_active:
            print("Detected active model from checkpoint. Using active diffusion model.")
            args.active = True

    inferred_dim = infer_base_dim_from_state(ckpt.get("model", {}))
    if inferred_dim is not None and inferred_dim != args.model_base_dim:
        print(f"Inferred base_dim {inferred_dim} from checkpoint. Overriding CLI value {args.model_base_dim}.")
        args.model_base_dim = inferred_dim

    dim_mults = _parse_int_list(args.dim_mults)
    attn_resolutions = tuple(_parse_int_list(args.attn_resolutions))

    device = torch.device(args.device)
    model = build_model(args, device, dim_mults, attn_resolutions)
    ema_model = load_ema_model(model, ckpt, device)
    generate_samples(ema_model, args)
    print(f"All samples saved to {args.output_dir}.")


if __name__ == "__main__":
    main()


