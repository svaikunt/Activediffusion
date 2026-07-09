import os
import math
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.utils import save_image
import inspect

from model_cifar10_sde_DDPM_v2 import CIFAR10Diffusion_SDE_V2, CIFAR10_Active_Diffusion_SDE_V2
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
    parser = argparse.ArgumentParser(description="CIFAR10 SDE Diffusion Training (v2 UNet shape)")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)

    # v2 architecture knobs
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2, help='UNet residual blocks per resolution (Table 6 uses 8)')
    parser.add_argument('--dim_mults', type=str, default="1,2,2,2", help='comma-separated UNet dim multipliers')
    parser.add_argument('--attn_resolutions', type=str, default="16,8", help='comma-separated resolutions for attention blocks')

    parser.add_argument('--timesteps', type=int, help='sampling steps of SDE', default=1000)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.999)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequency', default=10)
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--active', action='store_true', help='use active diffusion model', default=False)

    parser.add_argument('--Tp', type=float, default=1e-3, help='temperature parameter Tp')
    parser.add_argument('--Ta', type=float, default=1.0, help='active temperature Ta')
    parser.add_argument('--tau', type=float, default=0.4, help='persistence time tau')
    parser.add_argument('--k', type=float, default=1.0, help='spring constant k')
    parser.add_argument('--T', type=float, default=2.0, help='total time range T')

    parser.add_argument('--data_dir', type=str, default='./data', help='directory to store CIFAR-10 data')
    parser.add_argument('--save_freq', type=int, default=50, help='frequency (in epochs) to save checkpoints and samples')
    parser.add_argument('--large_sample_interval', type=int, default=50, help='epoch interval to save a large sample grid')
    parser.add_argument('--large_sample_count', type=int, default=100, help='number of images in the large sample grid')
    parser.add_argument('--pf_sample_interval', type=int, default=25, help='epoch interval to save probability-flow sample grid (0 disables)')
    parser.add_argument('--pf_sample_count', type=int, default=36, help='number of images in the PF sample grid')
    parser.add_argument('--pf_steps', type=int, default=0, help='ODE steps for probability-flow sampling (0 uses default timesteps)')
    parser.add_argument('--pf_schedule', type=str, default='linear', choices=['linear', 'log'], help='time discretization for PF sampling')
    parser.add_argument('--pf_solver', type=str, default='heun', choices=['heun', 'rk45'], help='PF-ODE solver (heun fixed-step or rk45 adaptive)')

    parser.add_argument('--out_dir', type=str, default='results_cifar10_sde_v2', help='directory to save checkpoints and samples')

    # keep these consistent with the main script (hygiene knobs)
    parser.add_argument('--lr_schedule', type=str, default='onecycle', choices=['onecycle', 'plateau_cosine'])
    parser.add_argument('--plateau_start_epoch', type=int, default=0)
    parser.add_argument('--plateau_end_epoch', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--warmup_lr', type=float, default=None)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for AdamW (set to 0.0 for diffusion)')

    return parser.parse_args()


def get_cifar10_dataloader(data_dir, batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )


def main(args):
    device = "cpu" if args.cpu else "cuda"
    print(f"Using device: {device}")

    os.makedirs(args.data_dir, exist_ok=True)
    train_dataloader = get_cifar10_dataloader(args.data_dir, batch_size=args.batch_size, num_workers=4)
    print(f"CIFAR-10 dataset loaded. Batches: {len(train_dataloader)}")

    dim_mults = _parse_int_list(args.dim_mults)
    attn_resolutions = tuple(_parse_int_list(args.attn_resolutions))

    if args.active:
        print("Using CIFAR-10 active diffusion model (SDE-style, v2)")
        model = CIFAR10_Active_Diffusion_SDE_V2(
            timesteps=args.timesteps,
            image_size=32,
            time_embedding_dim=256,
            base_dim=args.model_base_dim,
            num_res_blocks=args.num_res_blocks,
            dim_mults=dim_mults,
            attn_resolutions=attn_resolutions,
            Tp=args.Tp,
            Ta=args.Ta,
            k=args.k,
            tau=args.tau,
            T=args.T
        ).to(device)
    else:
        print("Using CIFAR-10 standard diffusion model (SDE-style, v2)")
        model = CIFAR10Diffusion_SDE_V2(
            timesteps=args.timesteps,
            image_size=32,
            in_channels=3,
            time_embedding_dim=256,
            base_dim=args.model_base_dim,
            num_res_blocks=args.num_res_blocks,
            dim_mults=dim_mults,
            attn_resolutions=attn_resolutions,
            T=args.T,
            k=args.k,
            Tp=args.Tp,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    warmup_lr = args.warmup_lr if args.warmup_lr is not None else args.lr
    optimizer = AdamW(model.parameters(), lr=warmup_lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            for param_group in optimizer.param_groups:
                param_group["lr"] = min(args.lr, param_group.get("lr", args.lr))
                param_group["weight_decay"] = args.weight_decay
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    if start_epoch >= args.epochs:
        print(f"Checkpoint epoch ({start_epoch}) >= target epochs ({args.epochs}). Increase --epochs to continue training.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    steps_per_epoch = len(train_dataloader)
    global_steps = start_epoch * steps_per_epoch

    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, min(args.warmup_steps, total_steps - 1))
    plateau_start_step = max(0, int(args.plateau_start_epoch * steps_per_epoch))
    plateau_end_step = max(plateau_start_step, int(args.plateau_end_epoch * steps_per_epoch))
    plateau_end_step = min(plateau_end_step, total_steps)
    decay_steps = max(1, total_steps - plateau_end_step)

    def compute_lr(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            warmup_frac = float(step_idx + 1) / float(warmup_steps)
            return warmup_lr + (args.lr - warmup_lr) * warmup_frac

        if args.lr_schedule == "plateau_cosine":
            if step_idx < plateau_end_step:
                return args.lr
            t = step_idx - plateau_end_step
            cos_decay = 0.5 * (1 + math.cos(math.pi * min(t / float(decay_steps), 1.0)))
            return args.min_lr + (args.lr - args.min_lr) * cos_decay

        return args.lr

    current_lr = compute_lr(global_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_dataloader):
            images = images.to(device)

            if args.active:
                eta_0 = model.generate_eta0(images.shape[0], device=device)
                x = torch.cat([images, eta_0], dim=1)
                t = 1e-3 + (args.T - 1e-3) * torch.rand(images.shape[0], device=device)
                (x_t, eta_t), (F_x, F_eta), means, cov, noise = model(x, t)
                loss = model.diffusion_loss_active((F_x, F_eta), images, eta_0, t, noise=noise)
            else:
                noise = torch.randn_like(images).to(device)
                x_t, score, mean, std = model.forward(images, noise)
                true_score = -(x_t - mean) / (std ** 2)
                loss = torch.nn.functional.mse_loss(std * score, std * true_score)

            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            current_lr = compute_lr(global_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            epoch_loss += loss.item()

            if batch_idx % args.log_freq == 0:
                print(
                    f"Epoch[{epoch+1}/{args.epochs}],Step[{batch_idx}/{len(train_dataloader)}],"
                    f"loss:{loss.detach().cpu().item():.5f},lr:{optimizer.param_groups[0]['lr']:.5f}"
                )

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_steps": global_steps,
                "avg_loss": avg_loss
            }
            ckpt_path = f"{args.out_dir}/checkpoint_epoch_{epoch+1:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

            model_ema.eval()
            model.eval()
            with torch.no_grad():
                def sample_images(model_to_sample, count, probability_flow=False):
                    pf_kwargs = {}
                    if probability_flow:
                        pf_kwargs = {
                            "probability_flow": True,
                            "pf_steps": args.pf_steps if args.pf_steps > 0 else None,
                            "pf_schedule": args.pf_schedule,
                        }
                        if _sampling_supports_pf_solver(model_to_sample):
                            pf_kwargs["pf_solver"] = args.pf_solver
                    if args.active:
                        imgs, _ = model_to_sample.sampling(count, device=device, **pf_kwargs)
                        return imgs
                    return model_to_sample.sampling(count, device=device, **pf_kwargs)

                samples = sample_images(model_ema.module, args.n_samples)
                samples_path = f"{args.out_dir}/samples_epoch_{epoch+1:03d}.png"
                save_image(samples, samples_path, nrow=max(1, int(math.sqrt(args.n_samples))), padding=2)
                print(f"Samples saved: {samples_path}")

                raw_samples = sample_images(model, args.n_samples)
                raw_samples_path = f"{args.out_dir}/samples_epoch_{epoch+1:03d}_raw.png"
                save_image(raw_samples, raw_samples_path, nrow=max(1, int(math.sqrt(args.n_samples))), padding=2)
                print(f"Raw samples saved: {raw_samples_path}")

                if args.large_sample_interval > 0 and args.large_sample_count > 0 and (epoch + 1) % args.large_sample_interval == 0:
                    large_samples = sample_images(model_ema.module, args.large_sample_count)
                    large_path = f"{args.out_dir}/samples_epoch_{epoch+1:03d}_large.png"
                    save_image(large_samples, large_path, nrow=max(1, int(math.sqrt(args.large_sample_count))), padding=2)
                    print(f"Large sample grid saved: {large_path}")

                if args.pf_sample_interval > 0 and args.pf_sample_count > 0 and (epoch + 1) % args.pf_sample_interval == 0:
                    pf_samples = sample_images(model_ema.module, args.pf_sample_count, probability_flow=True)
                    pf_path = f"{args.out_dir}/samples_epoch_{epoch+1:03d}_pf.png"
                    save_image(pf_samples, pf_path, nrow=max(1, int(math.sqrt(args.pf_sample_count))), padding=2)
                    print(f"Probability-flow sample grid saved: {pf_path}")

    print("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)


