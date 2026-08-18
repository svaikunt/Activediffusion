"""Noise real images to t*, then reverse -- with eta matched vs eta shuffled.

Both arms start from the SAME (x_t*, eta_t*) except that the shuffled arm has its
eta permuted across the batch.  Shuffling preserves eta's marginal exactly and
destroys only its pairing with x, so any difference in what comes back is
attributable to the x-eta correlation alone.

  python capture_recover.py --ckpt .../checkpoint_epoch_1200.pt --tstar 0.5
"""
import argparse, torch, numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from model_cifar10_sde_DDPM_v2 import CIFAR10_Active_Diffusion_SDE_V2
from generate_samples_multigpu import load_ema_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", default="recover.npz")
p.add_argument("--data_dir", default="/home/svaikunt/CIFAR10/data")
p.add_argument("--n", type=int, default=49)
p.add_argument("--tstar", type=float, default=0.5)
p.add_argument("--pf_steps", type=int, default=500)
p.add_argument("--pf_schedule", default="quadratic")
p.add_argument("--frames", type=int, default=44)
p.add_argument("--tau", type=float, default=0.5)
p.add_argument("--Ta", type=float, default=6.4)
p.add_argument("--Tp", type=float, default=1e-3)
p.add_argument("--k", type=float, default=4.0)
p.add_argument("--T", type=float, default=2.0)
p.add_argument("--score_param", default="cond")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

torch.manual_seed(a.seed)
dev = "cuda"
net = load_ema_model(CIFAR10_Active_Diffusion_SDE_V2(
    timesteps=1000, image_size=32, time_embedding_dim=256, base_dim=128,
    dim_mults=[1, 2, 2, 2], attn_resolutions=[16], num_res_blocks=4,
    Tp=a.Tp, Ta=a.Ta, k=a.k, tau=a.tau, T=a.T, score_param=a.score_param).to(dev),
    torch.load(a.ckpt, map_location=dev), dev)

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,)*3, (0.5,)*3)])
ds = datasets.CIFAR10(root=a.data_dir, train=True, download=False, transform=tf)
x0 = torch.stack([ds[i][0] for i in range(a.n)]).to(dev)          # [N,3,32,32] in [-1,1]
n = a.n
print(f"x0 {tuple(x0.shape)}  range [{x0.min():.2f}, {x0.max():.2f}]", flush=True)

Tp_t = torch.tensor(a.Tp, device=dev)
Ta_t = torch.tensor(a.Ta, device=dev)
tau_t = torch.tensor(a.tau, device=dev)
scale = net.eta_scale_tensor.view(1, 1, 1, 1).to(dev)

with torch.no_grad():
    # ---- forward: noise (x0, eta0) to t*
    eta0 = net.generate_eta0(n, device=dev)
    tv = torch.full((n,), a.tstar, device=dev)
    mean_x, mean_eta = net.compute_mean(x0, eta0, tv)
    nz = net.generate_correlated_noise(net.compute_covariance(tv), num_channels=3)
    xe = torch.zeros(n, 6, 32, 32, device=dev)
    xe[:, [0, 2, 4]] = mean_x + nz[:, [0, 2, 4]]
    xe[:, [1, 3, 5]] = mean_eta + nz[:, [1, 3, 5]]

    perm = torch.randperm(n, device=dev)
    xe_s = xe.clone()
    xe_s[:, [1, 3, 5]] = xe[perm][:, [1, 3, 5]]        # eta shuffled, x untouched
    print(f"corr(x,eta) matched {torch.corrcoef(torch.stack([xe[:,[0,2,4]].ravel(), xe[:,[1,3,5]].ravel()]))[0,1]:.4f}"
          f"   shuffled {torch.corrcoef(torch.stack([xe_s[:,[0,2,4]].ravel(), xe_s[:,[1,3,5]].ravel()]))[0,1]:.4f}",
          flush=True)

    grid = net._build_pf_time_grid(a.pf_steps, a.pf_schedule, dev, start_time=a.tstar)
    nstep = len(grid) - 1
    keep = sorted(set(np.linspace(0, nstep, a.frames, dtype=int).tolist()))

    def reverse(state, tag):
        snaps, times = [], []
        for idx in tqdm(range(nstep), desc=tag):
            if idx in keep:
                snaps.append(state.to(torch.float16).cpu().numpy())
                times.append(float(grid[idx]))
            t_c, t_n = grid[idx], grid[idx + 1]
            dt = (t_c - t_n).clamp(min=1e-6)
            tt = torch.full((n,), t_c.item(), device=dev)
            mi = state.clone(); mi[:, [1, 3, 5]] = mi[:, [1, 3, 5]] / scale
            o = net._score_from_output(net.model(mi, net._normalize_time(tt)), tt)
            d1 = net._active_pf_drift(state, o, Tp_t, Ta_t, tau_t)
            xp = state + d1 * dt
            tn = torch.full((n,), t_n.item(), device=dev)
            mn = xp.clone(); mn[:, [1, 3, 5]] = mn[:, [1, 3, 5]] / scale
            o2 = net._score_from_output(net.model(mn, net._normalize_time(tn)), tn)
            d2 = net._active_pf_drift(xp, o2, Tp_t, Ta_t, tau_t)
            state = state + 0.5 * (d1 + d2) * dt
        state = net._active_tweedie_correction(state, grid[-1].item(), dev)
        snaps.append(state.to(torch.float16).cpu().numpy()); times.append(0.0)
        return np.stack(snaps), np.array(times, dtype=np.float32)

    tm, times = reverse(xe, "matched")
    tsh, _ = reverse(xe_s, "shuffled")

np.savez_compressed(a.out, orig=x0.cpu().numpy().astype(np.float16),
                    matched=tm, shuffled=tsh, times=times,
                    tstar=a.tstar, tau=a.tau, perm=perm.cpu().numpy(), ckpt=a.ckpt)
mse = lambda z: float(((z[-1][:, [0, 2, 4]].astype(np.float32) - x0.cpu().numpy())**2).mean())
print(f"\nfinal MSE to original   matched {mse(tm):.4f}   shuffled {mse(tsh):.4f}", flush=True)
print(f"wrote {a.out}", flush=True)
