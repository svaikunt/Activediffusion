"""What does the active noise buy you?  Two views of the same quantity.

The denoiser sees the pair (x_t, eta_t).  The x-noise it must remove is d_x with
variance M11; conditioning on d_eta reduces that to v_x = det/M22.  So

    v_x / M11 = 1 - corr^2(d_x, d_eta)

is the fraction of the x-noise that eta does NOT explain.  Small = eta is
informative.  This is the same v_x that whitens the loss.

Left : v_x/M11 against t, per tau.
Right: measured FID against the t-average of v_x/M11.

python plot_eta_information.py
"""
import numpy as np, torch, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
Tp, Ta, k, T = 1e-3, 6.4, 4.0, 2.0

def cov(t, tau):
    w = 1 / tau
    A = torch.tensor([[-k, 1.], [0., -w]])
    Q = torch.tensor([[2 * Tp, 0.], [0., 2 * Ta / tau ** 2]])
    Z = torch.zeros(2, 2)
    C = torch.cat([torch.cat([-A, Q], 1), torch.cat([Z, A.T], 1)], 0)
    E = torch.linalg.matrix_exp(C * torch.as_tensor(t).reshape(-1, 1, 1))
    S = E[:, 2:, 2:].transpose(-1, -2) @ E[:, :2, 2:]
    return S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]

def unexplained(t, tau):
    m11, m12, m22 = cov(t, tau)
    return (1 - m12 ** 2 / (m11 * m22)).numpy()

TAUS = [0.10, 0.20, 0.25, 0.40, 0.50, 1.00]
FID  = {0.10: 12.88, 0.20: 11.68, 0.25: 13.88, 0.40: 9.68, 0.50: 8.89}
PREPATCH = {0.10, 0.40}

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#d62728", "#1baf7a", "#eda100", "#9b5de5"]
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.7), dpi=200)
fig.subplots_adjust(left=0.062, right=0.985, top=0.775, bottom=0.135, wspace=0.235)

ts = torch.cat([torch.logspace(-3, -1, 220), torch.linspace(0.1, T, 260)])
bar = torch.linspace(1e-3, T, 4001)
means = {}
for tau, c in zip(TAUS, COLORS):
    y = unexplained(ts, tau)
    axL.semilogx(ts.numpy(), y, color=c, lw=2 if tau == 0.5 else 1.5,
                 alpha=1.0 if tau == 0.5 else 0.8, zorder=3)
    axL.annotate(f"τ={tau:g}", xy=(T, y[-1]), xytext=(6, 0), textcoords="offset points",
                 color=c, fontsize=9, fontweight="bold", va="center")
    means[tau] = float(np.trapezoid(unexplained(bar, tau), bar.numpy()) / (T - 1e-3))

axL.set_xlim(1e-3, T)
axL.set_ylim(0, 1.04)
axL.set_xlabel("diffusion time  t", color=INK2, fontsize=10, labelpad=6)
axL.set_ylabel("fraction of x-noise unexplained by η\n$v_x/M_{11}=1-\\mathrm{corr}^2(d_x,d_\\eta)$",
               color=INK2, fontsize=10, labelpad=6)
axL.axhline(1.0, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
axL.annotate("η tells you nothing about the x-noise", xy=(1.8, 1.0),
             xytext=(-4, -12), textcoords="offset points", color=MUTED,
             fontsize=8, ha="right")

for tau in TAUS:
    if tau not in FID:
        continue
    c = COLORS[TAUS.index(tau)]
    pre = tau in PREPATCH
    axR.plot([means[tau]], [FID[tau]], "o", ms=11, color="none" if pre else c,
             mec=c, mew=2.0, zorder=4)
    dx, dy, ha = (0, 13, "center")
    axR.annotate(f"τ={tau:g}" + ("  pre-patch" if pre else ""),
                 xy=(means[tau], FID[tau]), xytext=(dx, dy),
                 textcoords="offset points", color=c, fontsize=8.8,
                 fontweight="bold", ha=ha, va="bottom")
ok = [t for t in FID if t not in PREPATCH and t != 0.25]
axR.plot([means[t] for t in sorted(ok)], [FID[t] for t in sorted(ok)],
         color=AXIS, lw=1.2, ls=(0, (5, 3)), zorder=2)
axR.annotate("critical damping, kτ=1\nbreaks the trend",
             xy=(means[0.25] - 0.006, FID[0.25]), xytext=(-16, 2),
             textcoords="offset points", color=INK2, fontsize=8.3, linespacing=1.4,
             ha="right", va="center",
             arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
axR.set_xlabel("mean $v_x/M_{11}$ over t   —   less is better informed by η",
               color=INK2, fontsize=10, labelpad=6)
axR.set_ylabel("FID   (N=10,000, quadratic PF-300, epoch 1000)",
               color=INK2, fontsize=10, labelpad=6)
axR.set_xlim(0.27, 0.76)
axR.set_ylim(8.0, 15.2)

for ax in (axL, axR):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDC, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)

fig.text(0.062, 0.955, "What the active noise contributes",
         color=INK, fontsize=14, fontweight="bold", ha="left", va="top")
fig.text(0.062, 0.908,
         "The denoiser sees (x, η) together. Conditioning on the η-noise removes part of the "
         "x-noise it has to undo:  $v_x=\\det M/M_{22}$.",
         color=MUTED, fontsize=8.6, ha="left", va="top")
fig.text(0.062, 0.868,
         "At τ=0.5 that is 68% of the x-noise variance, averaged over t. Larger kτ ⇒ η more "
         "informative ⇒ better FID — except at kτ=1, where the drift matrix is defective.",
         color=MUTED, fontsize=8.6, ha="left", va="top")

fig.savefig("eta_information.png", facecolor=SURFACE)
print("wrote eta_information.png")
for tau in TAUS:
    print(f"  tau={tau:<5} mean v_x/M11 = {means[tau]:.4f}"
          + (f"   FID {FID[tau]}" if tau in FID else ""))
