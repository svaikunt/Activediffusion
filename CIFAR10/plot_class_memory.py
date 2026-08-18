"""Class recovery after noise-and-reverse, with eta matched vs shuffled.

Shuffling eta preserves its marginal and destroys only its pairing with x.  The
shuffled arm sits at the 10% chance floor at every t*, so every bit of
recoverable class identity is carried by the x-eta correlation.

python plot_class_memory.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TS    = np.array([0.40, 0.50, 0.60, 0.80, 1.00])
PIX   = np.array([0.793, 0.659, 0.513, 0.252, 0.077])
KEPT  = np.array([0.611, 0.451, 0.324, 0.176, 0.105])
SHUF  = np.array([0.109, 0.107, 0.115, 0.111, 0.100])
CHANCE = 0.10

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDC, AXIS = "#e1e0d9", "#c3c2b7"
C_OK, C_BAD, C_PIX = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=200)
fig.subplots_adjust(left=0.083, right=0.975, top=0.700, bottom=0.125)

ax.axhspan(0, CHANCE, color=GRIDC, alpha=0.55, zorder=0)
ax.axhline(CHANCE, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
ax.plot(TS, KEPT, "-o", color=C_OK, lw=2.2, ms=8, mec=SURFACE, mew=2, zorder=4)
ax.plot(TS, SHUF, "-s", color=C_BAD, lw=2.2, ms=7, mec=SURFACE, mew=2, zorder=4)
ax.plot(TS, PIX, "-^", color=C_PIX, lw=1.6, ms=6, mec=SURFACE, mew=1.6,
        alpha=0.85, zorder=3)

ax.annotate("η matched", xy=(TS[0], KEPT[0]), xytext=(8, 8), textcoords="offset points",
            color=C_OK, fontsize=10.5, fontweight="bold")
ax.annotate("η shuffled — at chance everywhere", xy=(0.425, SHUF[0]), xytext=(6, 13),
            textcoords="offset points", color=C_BAD, fontsize=10, fontweight="bold")
ax.annotate("median pixel correlation\n(for reference, different scale)",
            xy=(TS[1], PIX[1]), xytext=(14, 16), textcoords="offset points",
            color=C_PIX, fontsize=8.6, linespacing=1.4)
ax.annotate("chance, 10 classes", xy=(0.355, CHANCE), xytext=(3, -13),
            textcoords="offset points", color=MUTED, fontsize=8, ha="left")

ax.set_xlim(0.34, 1.06)
ax.set_ylim(0, 0.88)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0", "20%", "40%", "60%", "80%"])
ax.set_xlabel("t*  —  how far the image was noised before reversing",
              color=INK2, fontsize=10, labelpad=6)
ax.set_ylabel("fraction recovering the original's class", color=INK2, fontsize=10, labelpad=6)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRIDC, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)

fig.text(0.083, 0.968, "Class identity is carried by the x–η correlation",
         color=INK, fontsize=14, fontweight="bold", ha="left", va="top")
fig.text(0.083, 0.922,
         "512 CIFAR-10 images noised to t*, reversed, then classified. Reference is the "
         "classifier's own prediction on the",
         color=MUTED, fontsize=8.6, ha="left", va="top")
fig.text(0.083, 0.882,
         "original, so classifier error cancels. Shuffling η across the batch leaves its "
         "distribution intact and breaks only the pairing.",
         color=MUTED, fontsize=8.6, ha="left", va="top")
fig.text(0.083, 0.836,
         "Shuffled sits at chance at every t*, so the pairing carries all of the recoverable "
         "class identity. At t*=0.5–0.6 the images come back",
         color=INK2, fontsize=8.6, ha="left", va="top")
fig.text(0.083, 0.802,
         "visibly different — pixel correlation 0.51–0.66 — yet a third to a half keep the "
         "right class.",
         color=INK2, fontsize=8.6, ha="left", va="top")
fig.text(0.083, 0.766,
         "Class does not outlive pixels here: both reach their floor at t*≈1.",
         color=INK2, fontsize=8.6, ha="left", va="top")

fig.savefig("class_memory.png", facecolor=SURFACE)
print("wrote class_memory.png")
