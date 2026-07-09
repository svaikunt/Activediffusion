import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion, MNIST_Active_Diffusion, MNIST_Linear_Diffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
import time
from datetime import timedelta

def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0,1] to [-1,1]
    ])

    train_dataset = MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess
    )
    test_dataset = MNIST(
        root="./mnist_data",
        train=False,
        download=True,
        transform=preprocess
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),\
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)    
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--restart', type=str, choices=['yes', 'no'], default='no', 
                        help='restart training from a checkpoint')
    parser.add_argument('--restart_epoch', type=int, default=0, 
                        help='epoch number to restart from (used with --restart=yes)')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of SDE', default=1000)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence', default=10)
    parser.add_argument('--save_freq', type=int, help='checkpoint saving frequency in epochs', default=5)
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--model_type', type=str, choices=['active', 'passive', 'linear'], default='active', 
                        help='model type: active, passive, or linear (arbitrary M, D) diffusion')
    parser.add_argument('--total_time', type=float, default=2.0, help='total time range for both active and passive models')
    # Passive model args
    parser.add_argument('--T', type=float, default=1.0, help='temperature parameter for passive diffusion')
    # Active model args
    parser.add_argument('--Tp', type=float, default=0.001, help='temperature parameter Tp for active diffusion')
    parser.add_argument('--Ta', type=float, default=1.0, help='active temperature Ta')
    parser.add_argument('--tau', type=float, default=0.5, help='persistence time tau')
    parser.add_argument('--k', type=float, default=1.0, help='spring constant k')
    # Linear model args
    parser.add_argument('--M', nargs=4, type=float, default=None,
                        metavar=('m11', 'm12', 'm21', 'm22'),
                        help='Drift matrix entries (linear model). dz = M z dt + D dW.')
    parser.add_argument('--D', nargs=4, type=float, default=None,
                        metavar=('d11', 'd12', 'd21', 'd22'),
                        help='Noise matrix entries (linear model).')
    parser.add_argument('--learn_M', action='store_true',
                        help='Make M a learnable parameter (linear model only).')
    parser.add_argument('--M_lr_factor', type=float, default=0.01,
                        help='Learning rate for M relative to main lr (default: 0.01).')
    parser.add_argument('--stability_weight', type=float, default=10.0,
                        help='Weight of the stability penalty when learn_M is True.')
    parser.add_argument('--M_l2_weight', type=float, default=0.01,
                        help='Weight of L2 (Frobenius norm) regularization on M.')
    parser.add_argument('--eta0_mode', type=str, default='stationary_marginal',
                        choices=['stationary_marginal', 'conditional', 'constant', 'zero'],
                        help='How to initialize eta_0 (linear model).')
    parser.add_argument('--eta0_value', type=float, default=0.0,
                        help='Constant value for eta_0 when eta0_mode=constant.')
    parser.add_argument('--name_suffix', type=str, default='linear',
                        help='Suffix used in checkpoint and sample filenames (linear model).')

    args = parser.parse_args()
    return args

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def write_stats_file(path, args, device, total_params, trainable_params,
                     epoch_stats, total_elapsed_seconds):
    hours, remainder = divmod(int(total_elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    total_time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    lines = []
    lines.append("=" * 60)
    lines.append("Training Run Summary")
    lines.append("=" * 60)
    lines.append(f"Model type       : {args.model_type} Diffusion")
    lines.append(f"Device           : {device}")
    lines.append(f"Total parameters : {total_params:,}")
    lines.append(f"Trainable params : {trainable_params:,}")
    lines.append("")

    lines.append("Hyperparameters:")
    lines.append(f"  lr             = {args.lr}")
    lines.append(f"  batch_size     = {args.batch_size}")
    lines.append(f"  epochs         = {args.epochs}")
    lines.append(f"  timesteps      = {args.timesteps}")
    lines.append(f"  total_time     = {args.total_time}")
    lines.append(f"  model_base_dim = {args.model_base_dim}")
    if args.model_type == 'active':
        lines.append(f"  Tp             = {args.Tp}")
        lines.append(f"  Ta             = {args.Ta}")
        lines.append(f"  tau            = {args.tau}")
        lines.append(f"  k              = {args.k}")
    elif args.model_type == 'passive':
        lines.append(f"  T              = {args.T}")
        lines.append(f"  k              = {args.k}")
    elif args.model_type == 'linear':
        lines.append(f"  M              = {args.M}")
        lines.append(f"  D              = {args.D}")
        lines.append(f"  learn_M        = {args.learn_M}")
        lines.append(f"  M_lr_factor    = {args.M_lr_factor}")
        lines.append(f"  stability_wt   = {args.stability_weight}")
        lines.append(f"  M_l2_weight    = {args.M_l2_weight}")
        lines.append(f"  eta0_mode      = {args.eta0_mode}")
    lines.append("")

    use_cuda = device.startswith("cuda")
    header = f"{'Epoch':>6}  {'Time (s)':>10}  {'Avg Loss':>12}"
    if use_cuda:
        header += f"  {'Peak GPU Mem (MB)':>18}"
    lines.append("Per-Epoch Statistics:")
    lines.append(header)
    lines.append("-" * (len(header) + 2))
    for s in epoch_stats:
        row = f"{s['epoch']:>6}  {s['epoch_time_s']:>10.2f}  {s['avg_loss']:>12.6f}"
        if use_cuda:
            row += f"  {s.get('peak_gpu_mb', 0.0):>18.1f}"
        if 'M_values' in s:
            row += f"  M={s['M_values']}"
        lines.append(row)
    lines.append("")
    lines.append(f"Total training time: {total_time_str}  ({total_elapsed_seconds:.1f} s)")
    lines.append("=" * 60)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Training stats saved to {path}")


def get_checkpoint_path(args, epoch):
    if args.model_type == 'active':
        return f"results/Active_{epoch}_Tp{args.Tp}_Ta{args.Ta}_tau{args.tau}.pt"
    elif args.model_type == 'linear':
        return f"results/Linear_{epoch}_{args.name_suffix}.pt"
    else:
        return f"results/Passive_{epoch}_T{args.T}_k{args.k}.pt"


def get_sample_path(args, epoch):
    if args.model_type == 'active':
        return f"results/Active_{epoch}_Tp{args.Tp}_Ta{args.Ta}_tau{args.tau}.png"
    elif args.model_type == 'linear':
        return f"results/Linear_{epoch}_{args.name_suffix}.png"
    else:
        return f"results/Passive_{epoch}_T{args.T}_k{args.k}.png"


def get_stats_path(args):
    if args.model_type == 'active':
        return f"results/Active_stats_Tp{args.Tp}_Ta{args.Ta}_tau{args.tau}.txt"
    elif args.model_type == 'linear':
        return f"results/Linear_stats_{args.name_suffix}.txt"
    else:
        return f"results/Passive_stats_T{args.T}_k{args.k}.txt"


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type} Diffusion")
    
    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=28
    )
    
    # ==================== Model creation ====================
    if args.model_type == 'active':
        model = MNIST_Active_Diffusion(
            timesteps=args.timesteps,
            image_size=28,
            time_embedding_dim=256,
            base_dim=args.model_base_dim,
            dim_mults=[2, 4],
            Tp=args.Tp,
            Ta=args.Ta,
            k=args.k,
            tau=args.tau,
            total_time=args.total_time
        ).to(device)
        print(f"Active model parameters: Tp={args.Tp}, Ta={args.Ta}, tau={args.tau}, k={args.k}, total_time={args.total_time}")

    elif args.model_type == 'linear':
        if args.M is None or args.D is None:
            raise ValueError("Both --M and --D are required for --model_type linear")
        model = MNIST_Linear_Diffusion(
            image_size=28,
            M_init=args.M,
            D=args.D,
            learn_M=args.learn_M,
            stability_weight=args.stability_weight,
            M_l2_weight=args.M_l2_weight,
            eta0_mode=args.eta0_mode,
            eta0_value=args.eta0_value,
            time_embedding_dim=256,
            timesteps=args.timesteps,
            base_dim=args.model_base_dim,
            dim_mults=[2, 4],
            total_time=args.total_time,
        ).to(device)
        print(f"Linear model: M={args.M}, D={args.D}, learn_M={args.learn_M}, "
              f"eta0_mode={args.eta0_mode}, total_time={args.total_time}")
        if args.learn_M:
            print(f"  M learning rate factor: {args.M_lr_factor}, "
                  f"stability weight: {args.stability_weight}, "
                  f"M L2 weight: {args.M_l2_weight}")

    else:  # passive
        model = MNISTDiffusion(
            timesteps=args.timesteps,
            image_size=28,
            in_channels=1,
            base_dim=args.model_base_dim,
            dim_mults=[2, 4],
            T=args.T,
            k=args.k,
            total_time=args.total_time
        ).to(device)
        print(f"Passive model parameters: T={args.T}, k={args.k}, total_time={args.total_time}, timesteps={args.timesteps}")
    
    print(f"Model timesteps: {args.timesteps}, dt: {model.dt:.6f}")

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")

    # torchvision ema setting
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    # ==================== Optimizer with separate M learning rate ====================
    if args.model_type == 'linear' and args.learn_M:
        # Separate parameter groups: small lr for M, normal lr for the U-Net
        unet_params = list(model.model.parameters())
        optimizer = AdamW([
            {'params': unet_params, 'lr': args.lr},
            {'params': [model.M], 'lr': args.lr * args.M_lr_factor},
        ])
        print(f"Optimizer: U-Net lr={args.lr}, M lr={args.lr * args.M_lr_factor}")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    scheduler = OneCycleLR(
        optimizer, 
        args.lr, 
        total_steps=args.epochs*len(train_dataloader),
        pct_start=0.25,
        anneal_strategy='cos'
    )
    
    loss_fn = nn.MSELoss(reduction='mean')

    # Initialize variables for tracking
    global_steps = 0
    start_epoch = 0
    
    # ==================== Checkpoint loading ====================
    if args.restart == 'yes' and args.restart_epoch > 0:
        restart_ckpt_path = get_checkpoint_path(args, args.restart_epoch)
        
        if os.path.exists(restart_ckpt_path):
            print(f"Restarting training from epoch {args.restart_epoch}")
            print(f"Loading checkpoint: {restart_ckpt_path}")
            
            ckpt = torch.load(restart_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            model_ema.load_state_dict(ckpt["model_ema"])
            
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("Optimizer state loaded")
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("Scheduler state loaded")
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
                print(f"Resuming from epoch {start_epoch}")
            if "global_steps" in ckpt:
                global_steps = ckpt["global_steps"]
                print(f"Resuming from global step {global_steps}")
            if "args" in ckpt:
                saved_args = ckpt["args"]
                print("\nOriginal training parameters:")
                for key, value in saved_args.items():
                    if key in vars(args) and vars(args)[key] != value:
                        print(f"  {key}: saved={value}, current={vars(args)[key]}")
            
            print(f"Successfully loaded checkpoint from epoch {args.restart_epoch}")
        else:
            print(f"Warning: Checkpoint file {restart_ckpt_path} not found. Starting from scratch.")
    
    elif args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        try:
            model_ema.load_state_dict(ckpt["model_ema"])
            model.load_state_dict(ckpt["model"])
            print("Model states loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading model states: {e}")
            from collections import OrderedDict
            if "model" in ckpt:
                new_state_dict = OrderedDict()
                for k, v in ckpt["model"].items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                print("Model loaded with key adjustments")
            if "model_ema" in ckpt:
                new_ema_state_dict = OrderedDict()
                for k, v in ckpt["model_ema"].items():
                    name = k[7:] if k.startswith('module.') else k
                    new_ema_state_dict[name] = v
                model_ema.load_state_dict(new_ema_state_dict)
                print("EMA model loaded with key adjustments")
    
    os.makedirs("results", exist_ok=True)

    epoch_stats = []
    train_start_time = time.perf_counter()

    # ==================== Training loop ====================
    for i in range(start_epoch, args.epochs):
        model.train()
        print(f"\nStarting epoch {i+1}/{args.epochs}")
        epoch_start_time = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for j, (image, target) in enumerate(train_dataloader):
            image = image.to(device)
            
            # ---- Compute loss depending on model type ----
            if args.model_type == 'active':
                eta_0 = model.generate_eta0(image.shape[0], device=device)
                x = torch.cat([image, eta_0], dim=1)
                t = 1e-3 + (args.total_time - 1e-3) * torch.rand(image.shape[0], device=device)
                (x_t, eta_t), (F_x, F_eta), means, cov, noise = model(x, t)
                loss = model.diffusion_loss_active((F_x, F_eta), image, eta_0, t, noise=noise)

            elif args.model_type == 'linear':
                eta_0 = model.generate_eta0(image.shape[0], device=device, x_0=image)
                x = torch.cat([image, eta_0], dim=1)
                t = 1e-3 + (args.total_time - 1e-3) * torch.rand(image.shape[0], device=device)
                (x_t, eta_t), eps_pred, eps, Q = model(x, t)
                loss = model.diffusion_loss(eps_pred, eps)

            else:  # passive
                noise = torch.randn_like(image).to(device)
                x_t, score, mean, std = model(image, noise)
                true_score = -(x_t - mean) / (std ** 2)
                loss = torch.nn.functional.mse_loss((std ** 2) * score, (std ** 2) * true_score)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss_sum += loss.detach().cpu().item()
            epoch_steps += 1

            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            
            if j % args.log_freq == 0:
                log_msg = (
                    f"Epoch[{i+1}/{args.epochs}],Step[{j}/{len(train_dataloader)}],"
                    f"loss:{loss.detach().cpu().item():.5f},lr:{scheduler.get_last_lr()[0]:.5f}"
                )
                if args.model_type == 'linear' and args.learn_M:
                    M_val = model.M.data.cpu().numpy()
                    tr_val = model.M.data.trace().item()
                    det_val = torch.det(model.M.data).item()
                    log_msg += (
                        f",M=[{M_val[0,0]:.3f},{M_val[0,1]:.3f};"
                        f"{M_val[1,0]:.3f},{M_val[1,1]:.3f}]"
                        f",tr={tr_val:.3f},det={det_val:.3f}"
                    )
                print(log_msg)

        # ---- End of epoch bookkeeping ----
        epoch_elapsed = time.perf_counter() - epoch_start_time
        avg_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else float("nan")
        stat = {
            "epoch": i + 1,
            "epoch_time_s": epoch_elapsed,
            "avg_loss": avg_loss,
        }
        if device.startswith("cuda"):
            stat["peak_gpu_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if args.model_type == 'linear' and args.learn_M:
            stat["M_values"] = model.M.data.cpu().numpy().round(4).tolist()
        epoch_stats.append(stat)

        summary_msg = (
            f"Epoch {i+1} complete — time: {epoch_elapsed:.1f}s, "
            f"avg loss: {avg_loss:.6f}"
        )
        if device.startswith("cuda"):
            summary_msg += f", peak GPU mem: {stat['peak_gpu_mb']:.1f} MB"
        if args.model_type == 'linear' and args.learn_M:
            M_val = model.M.data.cpu().numpy()
            summary_msg += (
                f"\n  Learned M = [{M_val[0,0]:.4f}, {M_val[0,1]:.4f}; "
                f"{M_val[1,0]:.4f}, {M_val[1,1]:.4f}]"
                f"  tr={model.M.data.trace().item():.4f}"
                f"  det={torch.det(model.M.data).item():.4f}"
            )
        print(summary_msg)

        # ---- Save checkpoint ----
        if (i + 1) % args.save_freq == 0 or (i + 1) == args.epochs:
            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": i + 1,
                "global_steps": global_steps,
                "args": vars(args)
            }
            checkpoint_path = get_checkpoint_path(args, i + 1)
            torch.save(ckpt, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            torch.save(ckpt, f"results/steps_{global_steps:0>8}.pt")

        # ---- Generate samples ----
        model_ema.eval()
        if args.model_type == 'active':
            x_samples, eta_samples = model_ema.module.sampling(args.n_samples, device=device)
            samples = x_samples
            eta_overall_mean = eta_samples.mean().item()
            eta_overall_std = eta_samples.std().item()
            print(f"Epoch {i+1}: eta_samples overall mean = {eta_overall_mean:.6f}, overall std = {eta_overall_std:.6f}")
            eta_per_dim_mean = eta_samples.mean(dim=0)
            eta_per_dim_std = eta_samples.std(dim=0)
            print(f"Epoch {i+1}: eta_samples per-dim mean shape = {eta_per_dim_mean.shape}, min = {eta_per_dim_mean.min().item():.6f}, max = {eta_per_dim_mean.max().item():.6f}")
            print(f"Epoch {i+1}: eta_samples per-dim std shape = {eta_per_dim_std.shape}, min = {eta_per_dim_std.min().item():.6f}, max = {eta_per_dim_std.max().item():.6f}")

        elif args.model_type == 'linear':
            x_samples, eta_samples = model_ema.module.sampling(args.n_samples, device=device)
            samples = x_samples
            print(f"Epoch {i+1}: eta mean={eta_samples.mean().item():.6f}, "
                  f"std={eta_samples.std().item():.6f}")

        else:  # passive
            samples = model_ema.module.sampling(args.n_samples, device=device)
        
        sample_path = get_sample_path(args, i + 1)
        save_image(samples, sample_path, nrow=int(math.sqrt(args.n_samples)))
        print(f"Samples saved to {sample_path}")
        save_image(samples, f"results/steps_{global_steps:0>8}.png",
                   nrow=int(math.sqrt(args.n_samples)))

    # ---- Final stats ----
    total_elapsed = time.perf_counter() - train_start_time
    stats_path = get_stats_path(args)
    write_stats_file(
        stats_path, args, device,
        total_params, trainable_params,
        epoch_stats, total_elapsed
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)