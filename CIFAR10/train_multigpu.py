"""
Multi-GPU training script for Active Diffusion on CIFAR-10
Works with: torchrun --standalone --nproc_per_node=N train_multigpu.py [args]
"""
import os
import math
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    parser = argparse.ArgumentParser(description="Multi-GPU CIFAR10 Active Diffusion Training")
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=256, help='Per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--ckpt', type=str, help='checkpoint path to resume from', default='')
    parser.add_argument('--n_samples', type=int, help='sample images after each save', default=36)

    # Architecture
    parser.add_argument('--model_base_dim', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=4, help='ResNet blocks per resolution')
    parser.add_argument('--dim_mults', type=str, default="1,2,2,2", help='UNet dim multipliers')
    parser.add_argument('--attn_resolutions', type=str, default="16,8", help='Attention resolutions')

    # Diffusion params
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--model_ema_steps', type=int, default=10)
    parser.add_argument('--model_ema_decay', type=float, default=0.999)
    parser.add_argument('--log_freq', type=int, default=10)
    
    # Model type
    parser.add_argument('--active', action='store_true', help='Use active diffusion')

    # Active diffusion parameters
    parser.add_argument('--Tp', type=float, default=1e-3)
    parser.add_argument('--Ta', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=2.0)

    # Directories
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./results_multigpu')
    
    # Checkpointing and sampling
    parser.add_argument('--save_freq', type=int, default=25)
    parser.add_argument('--large_sample_interval', type=int, default=50)
    parser.add_argument('--large_sample_count', type=int, default=100)
    parser.add_argument('--pf_sample_interval', type=int, default=50)
    parser.add_argument('--pf_sample_count', type=int, default=36)
    parser.add_argument('--pf_steps', type=int, default=0)
    parser.add_argument('--pf_schedule', type=str, default='linear', choices=['linear', 'log'])
    parser.add_argument('--pf_solver', type=str, default='heun', choices=['heun', 'rk45'])

    # Training hyperparams
    parser.add_argument('--score_param', type=str, default='score', choices=['score', 'whitened'],
                        help="Active model only. 'score': net output is the score (original). "
                             "'whitened': net predicts Sigma^{-1/2}(z-mu), unit-scale target at "
                             "every t. Changes what the network represents -- not resumable across.")
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    # Mixed precision
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (bf16)')

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if environment variables are set"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def get_dataloader(data_dir, batch_size, distributed=False, num_workers=4):
    """Create CIFAR-10 dataloader with optional distributed sampler"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    return loader, sampler


def main():
    args = parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main:
        print(f"{'='*60}")
        print(f"Multi-GPU Training Setup")
        print(f"{'='*60}")
        print(f"Distributed: {is_distributed}")
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Device: {device}")
        print(f"Active Diffusion: {args.active}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Num res blocks: {args.num_res_blocks}")
        print(f"Mixed precision: {args.amp}")
        print(f"{'='*60}\n")
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    
    # Create dataloader
    os.makedirs(args.data_dir, exist_ok=True)
    train_loader, train_sampler = get_dataloader(
        args.data_dir, 
        args.batch_size, 
        distributed=is_distributed
    )
    
    if is_main:
        print(f"Dataset loaded: {len(train_loader)} batches per GPU\n")
    
    # Parse architecture params
    dim_mults = _parse_int_list(args.dim_mults)
    attn_resolutions = tuple(_parse_int_list(args.attn_resolutions))
    
    # Create model
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
            score_param=args.score_param
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
    
    model = model.to(device)
    
    if is_main:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}\n")
    
    # Wrap with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    base_model = model.module if is_distributed else model
    
    # Setup EMA
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(base_model, device=device, decay=1.0 - alpha)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.ckpt and os.path.exists(args.ckpt):
        if is_main:
            print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        base_model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"Resumed from epoch {start_epoch}\n")
    
    # Create output directory
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Training loop
    steps_per_epoch = len(train_loader)
    global_step = start_epoch * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    
    # Learning rate schedule (simple linear warmup)
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr
    
    if is_main:
        print("Starting training...\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Update learning rate
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass with optional AMP
            if args.amp and torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if args.active:
                        eta_0 = base_model.generate_eta0(images.shape[0], device=device)
                        x = torch.cat([images, eta_0], dim=1)
                        t = 1e-3 + (args.T - 1e-3) * torch.rand(images.shape[0], device=device)
                        (x_t, eta_t), (F_x, F_eta), means, cov, noise = model(x, t)
                        loss = base_model.diffusion_loss_active((F_x, F_eta), images, eta_0, t, noise=noise)
                    else:
                        noise = torch.randn_like(images)
                        x_t, score, mean, std = model.forward(images, noise)
                        true_score = -(x_t - mean) / (std ** 2)
                        loss = torch.nn.functional.mse_loss(std * score, std * true_score)
            else:
                if args.active:
                    eta_0 = base_model.generate_eta0(images.shape[0], device=device)
                    x = torch.cat([images, eta_0], dim=1)
                    t = 1e-3 + (args.T - 1e-3) * torch.rand(images.shape[0], device=device)
                    (x_t, eta_t), (F_x, F_eta), means, cov, noise = model(x, t)
                    loss = base_model.diffusion_loss_active((F_x, F_eta), images, eta_0, t, noise=noise)
                else:
                    noise = torch.randn_like(images)
                    x_t, score, mean, std = model.forward(images, noise)
                    true_score = -(x_t - mean) / (std ** 2)
                    loss = torch.nn.functional.mse_loss((std ** 2) * score, (std ** 2) * true_score)
            
            # Backward pass
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if global_step % args.model_ema_steps == 0:
                model_ema.update_parameters(base_model)
            
            global_step += 1
            epoch_loss += loss.item()
            
            # Logging
            if batch_idx % args.log_freq == 0 and is_main:
                print(f"Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.5f} "
                      f"LR: {lr:.6f}")
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        
        # Reduce loss across GPUs for accurate logging
        if is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{args.epochs}] Complete")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"{'='*60}\n")
        
        # Save checkpoint and samples
        if is_main and ((epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1):
            # Save checkpoint
            ckpt = {
                "model": base_model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args)
            }
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            
            # Generate samples
            model.eval()
            model_ema.eval()
            
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
                
                # EMA samples
                samples = sample_images(model_ema.module, args.n_samples)
                sample_path = os.path.join(args.out_dir, f"samples_epoch_{epoch+1:03d}.png")
                save_image(samples, sample_path, nrow=int(math.sqrt(args.n_samples)), padding=2)
                print(f"Samples saved: {sample_path}")
                
                # Raw model samples
                raw_samples = sample_images(base_model, args.n_samples)
                raw_sample_path = os.path.join(args.out_dir, f"samples_epoch_{epoch+1:03d}_raw.png")
                save_image(raw_samples, raw_sample_path, nrow=int(math.sqrt(args.n_samples)), padding=2)
                print(f"Raw samples saved: {raw_sample_path}")
                
                # Large sample grid (periodic)
                if args.large_sample_interval > 0 and args.large_sample_count > 0 and (epoch + 1) % args.large_sample_interval == 0:
                    large_samples = sample_images(model_ema.module, args.large_sample_count)
                    large_path = os.path.join(args.out_dir, f"samples_epoch_{epoch+1:03d}_large.png")
                    save_image(large_samples, large_path, nrow=int(math.sqrt(args.large_sample_count)), padding=2)
                    print(f"Large sample grid saved: {large_path}")
                
                # Probability-flow samples (periodic)
                if args.pf_sample_interval > 0 and args.pf_sample_count > 0 and (epoch + 1) % args.pf_sample_interval == 0:
                    pf_samples = sample_images(model_ema.module, args.pf_sample_count, probability_flow=True)
                    pf_path = os.path.join(args.out_dir, f"samples_epoch_{epoch+1:03d}_pf.png")
                    save_image(pf_samples, pf_path, nrow=int(math.sqrt(args.pf_sample_count)), padding=2)
                    print(f"Probability-flow sample grid saved: {pf_path}")
                
            print()  # Extra newline for readability
    
    if is_main:
        print("Training complete!")
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

