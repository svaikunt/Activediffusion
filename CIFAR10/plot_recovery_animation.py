"""Noise a real image to t*, reverse, and see whether it comes back.

Three columns, all from the SAME noised state (x_t*, eta_t*):
  original          the images we started from
  matched eta       reverse using the eta that was actually paired with x
  shuffled eta      reverse with eta permuted across the batch

Shuffling preserves eta's marginal exactly -- same distribution, same amplitude --
and destroys only its pairing with x.  So the gap between columns 2 and 3 is
attributable to the x-eta correlation and nothing else.

eta_0 is drawn independently of x_0, so eta carries no information about the
*image*.  What it carries is information about the *noise that was added*, which
is what lets the model subtract the contamination instead of guessing at it.

  python plot_recovery_animation.py DownloadsFromCluster/recover_t0.20.npz
"""
import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--out", default="active_recovery.gif")
ap.add_argument("--grid", type=int, default=7)
ap.add_argument("--fps", type=int, default=11)
ap.add_argument("--hold", type=int, default=18, help="frames held on the final state")
ap.add_argument("--lead", type=int, default=10, help="frames held on the noised start")
ap.add_argument("--ease", type=int, default=8,
                help="show each of the first N frames twice, easing in")
ap.add_argument("--colors", type=int, default=72)
a = ap.parse_args()

d = np.load(a.npz)
orig = d["orig"].astype(np.float32)
mat = d["matched"].astype(np.float32)
shu = d["shuffled"].astype(np.float32)
times = d["times"]
tstar, tau = float(d["tstar"]), float(d["tau"])
G = a.grid; NG = G * G
orig = orig[:NG]
xm = mat[:, :NG, [0, 2, 4]]
xs = shu[:, :NG, [0, 2, 4]]
F, N, C, H, W = xm.shape
v0 = orig.var()

def tile(b):
    b = b.reshape(G, G, C, H, W).transpose(0, 3, 1, 4, 2)
    return b.reshape(G * H, G * W, C)

def show(b):
    return np.clip((tile(b) + 1) / 2, 0, 1)

mse_m = np.array([((xm[i] - orig) ** 2).mean() for i in range(F)]) / v0
mse_s = np.array([((xs[i] - orig) ** 2).mean() for i in range(F)]) / v0

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, AXIS = "#e1e0d9", "#c3c2b7"
C_OK, C_BAD = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

PPX = G * 32
FS = PPX / 224.0
fig = plt.figure(figsize=(PPX / 0.288 / 100, PPX / 0.450 / 100), dpi=100)
axes, ims = [], []
for j, (lab, col) in enumerate((("original", INK),
                                ("recovered — η matched", C_OK),
                                ("recovered — η shuffled", C_BAD))):
    ax = fig.add_axes([0.028 + j * 0.324, 0.315, 0.288, 0.450])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(AXIS); s.set_linewidth(1.0)
    ax.set_title(lab, color=col, fontsize=9.5 * FS, fontweight="bold", pad=6)
    axes.append(ax)
im0 = axes[0].imshow(show(orig), interpolation="nearest")
imM = axes[1].imshow(show(xm[0]), interpolation="nearest")
imS = axes[2].imshow(show(xs[0]), interpolation="nearest")

axC = fig.add_axes([0.028, 0.125, 0.936, 0.125])
axC.semilogy(times, mse_m, color=AXIS, lw=1.1, zorder=2)
axC.semilogy(times, mse_s, color=AXIS, lw=1.1, zorder=2)
lm, = axC.semilogy(times[:1], mse_m[:1], color=C_OK, lw=2.0, zorder=3)
ls, = axC.semilogy(times[:1], mse_s[:1], color=C_BAD, lw=2.0, zorder=3)
pm, = axC.semilogy([times[0]], [mse_m[0]], "o", color=C_OK, ms=5 * FS,
                   mec=SURFACE, mew=1.4, zorder=4)
ps, = axC.semilogy([times[0]], [mse_s[0]], "o", color=C_BAD, ms=5 * FS,
                   mec=SURFACE, mew=1.4, zorder=4)
axC.set_xlim(tstar * 1.04, -tstar * 0.04)
axC.set_ylim(min(mse_m.min(), 0.01) * 0.6, max(mse_s.max(), 1) * 2.2)
axC.axhline(1.0, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
axC.set_facecolor(SURFACE)
axC.yaxis.grid(True, color=GRIDC, lw=0.7)
axC.set_axisbelow(True)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axC.spines[s].set_color(AXIS)
axC.set_yticks([0.01, 0.1, 1.0])
axC.set_yticklabels(["1%", "10%", "100%"])
axC.minorticks_off()
axC.tick_params(colors=MUTED, labelsize=7 * FS, length=0, pad=2)
axC.set_xlabel("diffusion time  t      (reverse runs t* → 0)",
               color=INK2, fontsize=8 * FS, labelpad=2)
axC.text(0.0, 1.06, "MSE to the original, ÷ Var(x₀)", transform=axC.transAxes,
         color=INK2, fontsize=8 * FS, va="bottom")
axC.text(0.995, 0.90, "unrelated image", transform=axC.transAxes,
         color=MUTED, fontsize=6.8 * FS, ha="right", va="top")

fig.text(0.028, 0.983, f"Does the image come back?   noise to t*={tstar:g}, then reverse",
         color=INK, fontsize=12 * FS, fontweight="bold", ha="left", va="top")
fig.text(0.028, 0.938,
         f"CIFAR-10 · τ={tau:g} · both arms start from the identical noised state; "
         "η is permuted across the batch in the right column only",
         color=MUTED, fontsize=7.4 * FS, ha="left", va="top")
fig.text(0.028, 0.902,
         "Shuffling leaves η's distribution untouched and destroys only its pairing with x, "
         "so the gap is the x–η correlation.",
         color=MUTED, fontsize=7.4 * FS, ha="left", va="top")
clock = fig.text(0.972, 0.983, "", color=INK, fontsize=10.5 * FS,
                 fontweight="bold", ha="right", va="top", family="monospace")
score = axC.text(1.0, 1.06, "", color=INK2, fontsize=8 * FS,
                 ha="right", va="bottom", family="monospace",
                 transform=axC.transAxes)

frames = []
order = ([0] * a.lead
         + [i for i in range(min(a.ease, F)) for _ in (0, 1)]
         + list(range(min(a.ease, F), F))
         + [F - 1] * a.hold)
for i in order:
    imM.set_data(show(xm[i])); imS.set_data(show(xs[i]))
    lm.set_data(times[:i+1], mse_m[:i+1]); ls.set_data(times[:i+1], mse_s[:i+1])
    pm.set_data([times[i]], [mse_m[i]]); ps.set_data([times[i]], [mse_s[i]])
    clock.set_text(f"t = {times[i]:.3f}")
    score.set_text(f"matched {mse_m[i]:6.1%}   shuffled {mse_s[i]:6.1%}")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    frames.append(Image.fromarray(buf).quantize(colors=a.colors, method=Image.MEDIANCUT))

frames[0].save(a.out, save_all=True, append_images=frames[1:],
               duration=int(1000 / a.fps), loop=0, optimize=True, disposal=2)
print(f"wrote {a.out}  {frames[0].size}  {len(frames)} frames  "
      f"{os.path.getsize(a.out)/1e6:.1f} MB")
print(f"  final: matched {mse_m[-1]:.2%}   shuffled {mse_s[-1]:.2%}  of Var(x0)")
