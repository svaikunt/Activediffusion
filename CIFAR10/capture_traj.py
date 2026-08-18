"""Capture (x, eta) along the reverse PF-ODE, for animation.

Mirrors the Heun branch of CIFAR10_Active_Diffusion_SDE.sampling() exactly and
records x_eta at evenly spaced steps.  Writes a float16 npz; render elsewhere.

  python capture_traj.py --ckpt .../checkpoint_epoch_1200.pt --out traj_1200.npz
"""
import argparse, torch, numpy as np
from tqdm import tqdm
from model_cifar10_sde_DDPM_v2 import CIFAR10_Active_Diffusion_SDE_V2
from generate_samples_multigpu import load_ema_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", default="traj.npz")
p.add_argument("--n", type=int, default=64)
p.add_argument("--pf_steps", type=int, default=500)
p.add_argument("--pf_schedule", default="quadratic")
p.add_argument("--frames", type=int, default=56)
p.add_argument("--focus", type=float, default=0.30,
               help="concentrate frames below this t; almost nothing happens above it")
p.add_argument("--context", type=int, default=8,
               help="frames spread over the quiet stretch T -> focus")
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
model = CIFAR10_Active_Diffusion_SDE_V2(
    timesteps=1000, image_size=32, time_embedding_dim=256, base_dim=128,
    dim_mults=[1, 2, 2, 2], attn_resolutions=[16], num_res_blocks=4,
    Tp=a.Tp, Ta=a.Ta, k=a.k, tau=a.tau, T=a.T, score_param=a.score_param).to(dev)
net = load_ema_model(model, torch.load(a.ckpt, map_location=dev), dev)
print(f"loaded {a.ckpt}  score_param={a.score_param}  tau={a.tau}", flush=True)

n = a.n
with torch.no_grad():
    base_cov = net.compute_covariance(a.T * torch.ones(n, device=dev))
    x_eta = net.generate_correlated_noise(base_cov, num_channels=3)
    Tp_t = torch.tensor(a.Tp, device=dev)
    Ta_t = torch.tensor(a.Ta, device=dev)
    tau_t = torch.tensor(a.tau, device=dev)
    grid = net._build_pf_time_grid(a.pf_steps, a.pf_schedule, x_eta.device, t_end=None)
    nstep = len(grid) - 1
    gnp = grid.detach().cpu().numpy()
    i_focus = int(np.searchsorted(-gnp, -a.focus))          # grid descends T -> 0
    n_det = max(2, a.frames - a.context)
    keep = sorted(set(np.linspace(0, i_focus, a.context, dtype=int).tolist()
                      + np.linspace(i_focus, nstep, n_det, dtype=int).tolist()))
    print(f"t<={a.focus} starts at step {i_focus}/{nstep}; "
          f"{a.context} context + {n_det} detail frames", flush=True)
    snaps, times = [], []

    scale = net.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)
    for idx in tqdm(range(nstep), desc="PF-ODE"):
        if idx in keep:
            snaps.append(x_eta.detach().to(torch.float16).cpu().numpy())
            times.append(float(grid[idx]))
        t_c, t_n = grid[idx], grid[idx + 1]
        dt = (t_c - t_n).clamp(min=1e-6)

        tt = torch.full((n,), t_c.item(), device=dev)
        mi = x_eta.clone(); mi[:, [1, 3, 5]] = mi[:, [1, 3, 5]] / scale
        out = net._score_from_output(net.model(mi, net._normalize_time(tt)), tt)
        d_c = net._active_pf_drift(x_eta, out, Tp_t, Ta_t, tau_t)

        x_pred = x_eta + d_c * dt
        tn = torch.full((n,), t_n.item(), device=dev)
        mn = x_pred.clone(); mn[:, [1, 3, 5]] = mn[:, [1, 3, 5]] / scale
        out_n = net._score_from_output(net.model(mn, net._normalize_time(tn)), tn)
        d_n = net._active_pf_drift(x_pred, out_n, Tp_t, Ta_t, tau_t)

        x_eta = x_eta + 0.5 * (d_c + d_n) * dt

    snaps.append(x_eta.detach().to(torch.float16).cpu().numpy())
    times.append(float(grid[-1]))
    x_eta = net._active_tweedie_correction(x_eta, grid[-1].item(), dev)
    snaps.append(x_eta.detach().to(torch.float16).cpu().numpy())
    times.append(0.0)

arr = np.stack(snaps)                      # [frames, n, 6, H, W]
np.savez_compressed(a.out, traj=arr, times=np.array(times, dtype=np.float32),
                    tau=a.tau, T=a.T, Ta=a.Ta, k=a.k, ckpt=a.ckpt)
print(f"wrote {a.out}  shape={arr.shape}  "
      f"{arr.nbytes/1e6:.0f} MB raw", flush=True)
