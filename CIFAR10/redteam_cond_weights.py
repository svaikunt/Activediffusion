import torch, math, sys
sys.path.insert(0, "/Users/suriyanarayanvaikuntanathan/Documents/Academics/Code/Active_Diffusion_final_code/CIFAR10")
from model_cifar10_sde_DDPM import CIFAR10_Active_Diffusion_SDE
torch.manual_seed(0)
m = CIFAR10_Active_Diffusion_SDE(score_param="cond", image_size=32, base_dim=128,
    dim_mults=[1,2,2,2], num_res_blocks=1, Tp=1e-3, Ta=6.4, k=4.0, tau=0.5,
    T=2.0, timesteps=1000).eval()

# A (redone): compare against the RMS scale of the target, not per-element magnitude
t = (1e-3 + (2.0-1e-3)*torch.rand(256)).double()
M = m.compute_covariance(t.float()).double()
noise = m.generate_correlated_noise(M.float(), num_channels=3).double()
dx, de = noise[:,[0,2,4]], noise[:,[1,3,5]]
M11=M[:,0,0].view(-1,1,1,1); M12=M[:,0,1].view(-1,1,1,1); M22=M[:,1,1].view(-1,1,1,1)
det = M11*M22 - M12*M12
sx_e = -(M22*dx - M12*de)/det ; se_e = -(M11*de - M12*dx)/det
vx = M11 - M12*M12/M22 ; ve = M22 - M12*M12/M11
sx_c = -(dx - (M12/M22)*de)/vx ; se_c = -(de - (M12/M11)*dx)/ve
for nm, a, b in (("x", sx_c, sx_e), ("eta", se_c, se_e)):
    print(f"A. {nm:3s}: max|diff|/RMS(target) = "
          f"{((a-b).abs().max()/b.pow(2).mean().sqrt()).item():.3e}")

# E: what the reweighting actually is -- cond weight / score weight
print("\nE. effective loss weight ratio  cond/score  (= 1/M_jj)")
print(f"   {'t':>7} {'M11':>11} {'M22':>11} {'x: 1/M22':>11} {'eta: 1/M11':>12}")
tt = torch.tensor([1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0]).double()
Mt = m.compute_covariance(tt)
for i, tv in enumerate(tt):
    a11, a22 = Mt[i,0,0].item(), Mt[i,1,1].item()
    print(f"   {tv.item():7.3f} {a11:11.4e} {a22:11.4e} {1/a22:11.4e} {1/a11:12.4e}")
r = [(1/Mt[i,0,0]).item() for i in range(len(tt))]
print(f"   eta-channel reweighting spans {max(r)/min(r):.3g}x across t")
