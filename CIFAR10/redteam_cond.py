"""Red-team the score_param='cond' loss against the original 'score' loss.

Question: did changing the whitener change WHAT the network learns (the dynamics),
or only HOW the residuals are weighted during training?
"""
import torch, math, sys
sys.path.insert(0, "/Users/suriyanarayanvaikuntanathan/Documents/Academics/Code/Active_Diffusion_final_code/CIFAR10")
from model_cifar10_sde_DDPM import CIFAR10_Active_Diffusion_SDE

torch.manual_seed(0)
KW = dict(image_size=32, base_dim=128, dim_mults=[1, 2, 2, 2], num_res_blocks=1,
          Tp=1e-3, Ta=6.4, k=4.0, tau=0.5, T=2.0, timesteps=1000)
m_sc = CIFAR10_Active_Diffusion_SDE(score_param="score", **KW).eval()
m_cd = CIFAR10_Active_Diffusion_SDE(score_param="cond",  **KW).eval()

B = 64
t = (1e-3 + (2.0 - 1e-3) * torch.rand(B)).float()
x0 = torch.randn(B, 3, 32, 32)
e0 = torch.randn(B, 3, 32, 32) * math.sqrt(6.4 / 0.5)

M = m_sc.compute_covariance(t)
noise = m_sc.generate_correlated_noise(M, num_channels=3)
dx, de = noise[:, [0, 2, 4]], noise[:, [1, 3, 5]]
M11 = M[:, 0, 0].view(-1, 1, 1, 1); M12 = M[:, 0, 1].view(-1, 1, 1, 1)
M22 = M[:, 1, 1].view(-1, 1, 1, 1); det = M11 * M22 - M12 * M12

# ---- A. is cond's target the exact score, -Sigma^{-1} d ?
s_x_exact = -(M22 * dx - M12 * de) / det
s_e_exact = -(M11 * de - M12 * dx) / det
v_x = M11 - M12 * M12 / M22
v_e = M22 - M12 * M12 / M11
s_x_cond = -(dx - (M12 / M22) * de) / v_x
s_e_cond = -(de - (M12 / M11) * dx) / v_e
rx = ((s_x_cond - s_x_exact).abs() / s_x_exact.abs().clamp(min=1e-30)).max().item()
re = ((s_e_cond - s_e_exact).abs() / s_e_exact.abs().clamp(min=1e-30)).max().item()
print(f"A. cond target == -Sigma^-1 d :  max rel err  x {rx:.3e}   eta {re:.3e}")

# ---- B. do both losses vanish at the exact score?  (same minimizer)
with torch.no_grad():
    l_sc = m_sc.diffusion_loss_active((s_x_exact, s_e_exact), x0, e0, t, noise=noise).item()
    l_cd = m_cd.diffusion_loss_active((s_x_exact, s_e_exact), x0, e0, t, noise=noise).item()
    l_sc0 = m_sc.diffusion_loss_active((torch.zeros_like(dx), torch.zeros_like(de)), x0, e0, t, noise=noise).item()
    l_cd0 = m_cd.diffusion_loss_active((torch.zeros_like(dx), torch.zeros_like(de)), x0, e0, t, noise=noise).item()
print(f"B. loss at exact score        :  score {l_sc:.3e}   cond {l_cd:.3e}")
print(f"   loss at zero output        :  score {l_sc0:.4f}      cond {l_cd0:.4f}")

# ---- C. float32 cancellation in v_x, v_e
t64 = t.double()
M64 = m_sc.compute_covariance(t64)
a11, a12, a22 = M64[:, 0, 0], M64[:, 0, 1], M64[:, 1, 1]
vx64 = a11 - a12 * a12 / a22
ve64 = a22 - a12 * a12 / a11
ex = ((v_x.view(-1).double() - vx64).abs() / vx64.abs()).max().item()
ee = ((v_e.view(-1).double() - ve64).abs() / ve64.abs()).max().item()
print(f"C. v float32 vs float64       :  max rel err  v_x {ex:.3e}   v_eta {ee:.3e}")
tt = torch.tensor([1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0])
Mt = m_sc.compute_covariance(tt); Mt64 = m_sc.compute_covariance(tt.double())
for i, tv in enumerate(tt):
    v32 = (Mt[i, 0, 0] - Mt[i, 0, 1] ** 2 / Mt[i, 1, 1]).item()
    v64 = (Mt64[i, 0, 0] - Mt64[i, 0, 1] ** 2 / Mt64[i, 1, 1]).item()
    rho = (Mt64[i, 0, 1] / (Mt64[i, 0, 0] * Mt64[i, 1, 1]).sqrt()).item()
    print(f"     t={tv:6.3f}  rho={rho:+.6f}  v_x32={v32:.6e}  v_x64={v64:.6e}"
          f"  rel={abs(v32 - v64) / abs(v64):.2e}")

# ---- D. is the sampling path identical?
tq = torch.rand(8) * 2
out = torch.randn(8, 6, 32, 32)
d_sc = m_sc._score_from_output(out, tq)
d_cd = m_cd._score_from_output(out, tq)
print(f"D. _score_from_output identity:  score {torch.equal(d_sc, out)}   "
      f"cond {torch.equal(d_cd, out)}   agree {torch.equal(d_sc, d_cd)}")
