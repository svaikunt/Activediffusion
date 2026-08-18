"""Animate reverse-time active sampling: the image x beside the active noise eta.

Fixed normalisation in both panels, so the amplitudes are real: x is clipped to
[-1, 1]; eta to +/-10, which is its 99.5th percentile at *every* t -- eta is an
autonomous OU process already at equilibrium, so its magnitude never changes
during sampling. What changes is its relationship to x.

That is the point of the pairing. At t = T the panels are near-copies of one
another (corr = 0.82, the stationary value). By t = 0 they are unrelated: x has
acquired image statistics and decoupled from the field that drove it.

The time axis is sqrt(t), because essentially nothing happens above t ~ 0.3.

  python plot_traj_animation.py DownloadsFromCluster/traj_1200_focus.npz
"""
import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--out", default="active_sampling.gif")
ap.add_argument("--grid", type=int, default=8)
ap.add_argument("--eta_clip", type=float, default=10.0)
ap.add_argument("--fps", type=int, default=12)
ap.add_argument("--hold", type=int, default=18)
ap.add_argument("--colors", type=int, default=96)
ap.add_argument("--fid", default="4.88")
a = ap.parse_args()

d = np.load(a.npz)
traj, times = d["traj"].astype(np.float32), d["times"]
tau = float(d["tau"]) if "tau" in d else 0.5
G = a.grid
x = traj[:, :G*G, [0, 2, 4]]
e = traj[:, :G*G, [1, 3, 5]]
F, N, C, H, W = x.shape

def tile(b):
    b = b.reshape(G, G, C, H, W).transpose(0, 3, 1, 4, 2)
    return b.reshape(G*H, G*W, C)

rho = np.array([np.corrcoef(x[i].ravel(), e[i].ravel())[0, 1] for i in range(F)])
st = np.sqrt(np.maximum(times, 0.0))                      # plotting coordinate

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, AXIS, ACC = "#e1e0d9", "#c3c2b7", "#2a78d6"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "figure.facecolor": SURFACE})

# size the figure so each panel renders at exactly G*32 native pixels
PPX = G * 32
FS = PPX / 256.0          # scale type with the figure
fig = plt.figure(figsize=(PPX / 0.371 / 100, PPX / 0.527 / 100), dpi=100)
axX = fig.add_axes([0.036, 0.243, 0.371, 0.527])
axE = fig.add_axes([0.593, 0.243, 0.371, 0.527])
axC = fig.add_axes([0.036, 0.092, 0.928, 0.104])

imX = axX.imshow(np.clip((tile(x[0]) + 1) / 2, 0, 1), interpolation="nearest")
imE = axE.imshow(np.clip((tile(e[0]) / a.eta_clip + 1) / 2, 0, 1), interpolation="nearest")
for ax, lab in ((axX, "x    image"), (axE, "η    active noise")):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(AXIS); s.set_linewidth(1.0)
    ax.set_title(lab, color=INK, fontsize=11*FS, fontweight="bold", pad=6)

axC.plot(st, rho, color=AXIS, lw=1.2, zorder=2)
seen, = axC.plot(st[:1], rho[:1], color=ACC, lw=2.0, zorder=3)
mk, = axC.plot([st[0]], [rho[0]], "o", color=ACC, ms=5.5*FS,
               mec=SURFACE, mew=1.5, zorder=4)
axC.set_xlim(st.max() + 0.05, -0.05)
axC.set_ylim(-0.14, 1.02)
axC.set_yticks([0, 0.4, 0.8])
ticks = [2.0, 1.0, 0.5, 0.2, 0.05, 0.0]
axC.set_xticks([np.sqrt(v) for v in ticks])
axC.set_xticklabels([f"{v:g}" for v in ticks])
axC.set_facecolor(SURFACE)
axC.yaxis.grid(True, color=GRIDC, lw=0.7)
axC.set_axisbelow(True)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axC.spines[s].set_color(AXIS)
axC.tick_params(colors=MUTED, labelsize=7*FS, length=0, pad=2)
axC.set_xlabel("diffusion time  t   (√t axis; integration runs T → 0)",
               color=INK2, fontsize=8*FS, labelpad=2)
axC.text(1.0, 1.06, "corr(x, η)", transform=axC.transAxes,
         color=INK2, fontsize=8*FS, va="bottom", ha="right")

fig.text(0.036, 0.984, "Active diffusion — reverse-time sampling",
         color=INK, fontsize=13*FS, fontweight="bold", ha="left", va="top")
fig.text(0.036, 0.938,
         f"CIFAR-10 · τ={tau:g} · PF-ODE 500 steps (quadratic, Heun) · FID {a.fid}",
         color=MUTED, fontsize=7.6*FS, ha="left", va="top")
fig.text(0.036, 0.901,
         f"fixed scaling: x clipped to [−1, 1], η to ±{a.eta_clip:g} — η is at equilibrium, "
         "its amplitude never changes",
         color=MUTED, fontsize=7.6*FS, ha="left", va="top")
clock = fig.text(0.964, 0.984, "", color=INK, fontsize=11*FS, fontweight="bold",
                 ha="right", va="top", family="monospace")
note = fig.text(0.964, 0.938, "", color=MUTED, fontsize=7.6*FS, ha="right", va="top")

frames = []
for i in list(range(F)) + [F - 1] * a.hold:
    imX.set_data(np.clip((tile(x[i]) + 1) / 2, 0, 1))
    imE.set_data(np.clip((tile(e[i]) / a.eta_clip + 1) / 2, 0, 1))
    seen.set_data(st[:i+1], rho[:i+1])
    mk.set_data([st[i]], [rho[i]])
    clock.set_text(f"t = {times[i]:.3f}")
    note.set_text("x and η are near-copies" if rho[i] > 0.6
                  else ("decoupling" if rho[i] > 0.15
                        else "x is an image; η is unrelated"))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    frames.append(Image.fromarray(buf).quantize(colors=a.colors, method=Image.MEDIANCUT))

frames[0].save(a.out, save_all=True, append_images=frames[1:],
               duration=int(1000 / a.fps), loop=0, optimize=True, disposal=2)
print(f"wrote {a.out}  {frames[0].size}  {len(frames)} frames  "
      f"{os.path.getsize(a.out)/1e6:.1f} MB")
