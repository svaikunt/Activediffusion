"""Active vs passive CIFAR-10 FID, under both PF schedules, with the config on the figure.

Colour = model (blue active / orange passive).
Line style = sampler schedule (solid = quadratic PF-500, dashed = log PF-600).

The quadratic series are the valid comparison: the active model is schedule-insensitive
(<0.2 FID) but the passive model is not (6-7 FID, widening), so the log curves understate
passive badly and are kept only for reference.

Edit the DATA block and re-run:  python plot_fid_comparison.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------- data
ACTIVE_LOG = [(1000, 21.58), (1200, 19.20), (1400, 18.24), (1600, 17.38),
              (1800, 16.50), (2000, 15.39), (2200, 14.03), (2400, 12.69),
              (2600, 12.34), (2800, 12.20), (4000, 13.04)]
PASSIVE_LOG = [(1000, 21.15), (1200, 20.64), (1400, 20.63), (1600, 21.01),
               (2200, 20.95)]           # 1800/2000 not measured
ACTIVE_QUAD = [(1000, 21.75), (1400, 18.08), (2200, 13.86)]
PASSIVE_QUAD = [(1000, 15.09), (1400, 14.40), (2200, 13.65), (2600, 14.41),
                (3600, 14.95)]

OUT = "fid_active_vs_passive.png"

# ---------------------------------------------------------------- palette
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
C_ACTIVE  = "#2a78d6"   # categorical slot 1
C_PASSIVE = "#eb6834"   # categorical slot 2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

fig, ax = plt.subplots(figsize=(10.6, 6.0), dpi=200)
fig.subplots_adjust(left=0.075, right=0.695, top=0.815, bottom=0.225)

# ---------------------------------------------------------------- chrome
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.xaxis.grid(False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
    ax.spines[side].set_linewidth(1.0)
ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
ax.xaxis.set_major_locator(MultipleLocator(400))

# ---------------------------------------------------------------- marks
def series(pts, color, marker, primary):
    """primary=True -> quadratic (solid, emphasised); False -> log (dashed, recessive)."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color,
            linewidth=2.0 if primary else 1.4,
            linestyle="-" if primary else (0, (5, 3)),
            alpha=1.0 if primary else 0.45,
            marker=marker, markersize=7 if primary else 5,
            markerfacecolor=color, markeredgecolor=SURFACE,
            markeredgewidth=2 if primary else 1.2,
            zorder=3 if primary else 2, clip_on=False)

series(ACTIVE_LOG,   C_ACTIVE,  "o", primary=False)
series(PASSIVE_LOG,  C_PASSIVE, "s", primary=False)
series(ACTIVE_QUAD,  C_ACTIVE,  "o", primary=True)
series(PASSIVE_QUAD, C_PASSIVE, "s", primary=True)

# direct labels — placed where each series is well separated, not all at the ends
ax.annotate("Active · quadratic", xy=(1400, 18.08), xytext=(10, 12),
            textcoords="offset points", color=C_ACTIVE, fontsize=10,
            fontweight="bold")
ax.annotate("Passive · quadratic", xy=(1400, 14.40), xytext=(8, 16),
            textcoords="offset points", color=C_PASSIVE, fontsize=10,
            fontweight="bold")
ax.annotate("Passive · log", xy=(2200, 20.95), xytext=(12, 0),
            textcoords="offset points", color=C_PASSIVE, fontsize=9.5,
            alpha=0.8, va="center")
ax.annotate("Active · log", xy=(2600, 12.34), xytext=(12, 0),
            textcoords="offset points", color=C_ACTIVE, fontsize=9.5,
            alpha=0.8, va="center")

# the crossover
ax.annotate("crossover ≈ 2200\n13.86 vs 13.65 — a tie",
            xy=(2210, 13.76), xytext=(26, 26), textcoords="offset points",
            color=INK_2, fontsize=8.5, ha="left", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))

# each model bottoms out then drifts back up — mark the minima
for xm, ym, lab, col in ((2800, 12.20, "active best 12.20 @ 2800", C_ACTIVE),
                         (2200, 13.65, "passive best 13.65 @ 2200", C_PASSIVE)):
    ax.axhline(ym, xmin=0.30, color=col, lw=1, ls=(0, (1, 3)), alpha=0.5, zorder=1)
    ax.annotate(lab, xy=(4330, ym), xytext=(0, 5), textcoords="offset points",
                color=col, fontsize=8, alpha=0.9, ha="right")

# ---------------------------------------------------------------- labels
ax.set_xlabel("training epoch", color=INK_2, fontsize=10, labelpad=8)
ax.set_ylabel("FID  (lower is better)", color=INK_2, fontsize=10, labelpad=8)
ax.set_xlim(960, 4380)
ax.set_ylim(11.3, 22.6)

fig.text(0.075, 0.965, "Active vs passive diffusion — CIFAR-10 FID",
         color=INK, fontsize=15, fontweight="bold", ha="left", va="top")
fig.text(0.075, 0.918,
         "Solid = quadratic PF-500 (matched sampler, the valid comparison) · "
         "dashed = log PF-600 (understates passive by 6–7 FID)",
         color=MUTED, fontsize=8.5, ha="left", va="top")
fig.text(0.075, 0.883,
         "Heun · Tweedie on · N = 10,000 · clean-fid (legacy_pytorch) vs CIFAR-10 train · "
         "within-plateau scatter ≈ 0.8 FID",
         color=MUTED, fontsize=8.5, ha="left", va="top")

handles = [
    Line2D([], [], color=C_ACTIVE, lw=2.0, marker="o", ms=7, mec=SURFACE, mew=2,
           label="Active · quadratic PF-500"),
    Line2D([], [], color=C_PASSIVE, lw=2.0, marker="s", ms=7, mec=SURFACE, mew=2,
           label="Passive · quadratic PF-500"),
    Line2D([], [], color=C_ACTIVE, lw=1.4, ls=(0, (5, 3)), alpha=0.45, marker="o",
           ms=5, mec=SURFACE, mew=1.2, label="Active · log PF-600"),
    Line2D([], [], color=C_PASSIVE, lw=1.4, ls=(0, (5, 3)), alpha=0.45, marker="s",
           ms=5, mec=SURFACE, mew=1.2, label="Passive · log PF-600"),
]
legend = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9,
                   handlelength=2.2, borderpad=0, labelspacing=0.5)
for t in legend.get_texts():
    t.set_color(INK_2)

# ---------------------------------------------------------------- config panel
CONFIG = [
    ("Architecture", ["UNet · base 128 · mults 1,2,2,2",
                      "4 res-blocks · attn @ 16",
                      "57.0M parameters"]),
    ("Optimization", ["AdamW · wd 0 · grad-clip 1.0",
                      "lr 1e-4 → 5e-5 at epoch 1000",
                      "warmup 5k steps, then flat",
                      "batch 4 × 128 = 512 · AMP"]),
    ("EMA",          ["epochs 1-1000:    0.998523",
                      "epochs 1000-:     0.99985",
                      "(nominal 0.997 / 0.9997,",
                      "torchvision-adjusted, every 10 steps)"]),
    ("SDE",          ["active:  Ta 6.4 · Tp 1e-3 · τ 0.15",
                      "passive: Tp 6.4",
                      "both:    k 4 · T 2 · 1000 steps"]),
]

X = 0.715
y = 0.845
HEAD_GAP, LINE, BLOCK_GAP = 0.042, 0.030, 0.033
for head, lines in CONFIG:
    fig.text(X, y, head, color=INK, fontsize=8.8, fontweight="bold",
             ha="left", va="top")
    y -= HEAD_GAP
    for line in lines:
        fig.text(X, y, line, color=MUTED, fontsize=7.8, ha="left", va="top")
        y -= LINE
    y -= BLOCK_GAP

fig.text(0.075, 0.105,
         "Both runs share identical architecture, optimizer, lr schedule, effective batch and EMA;\n"
         "they differ only in the SDE. Passive leads early and bottoms at 13.65 (epoch 2200); active\n"
         "starts 6.7 behind, improves ~5x faster, bottoms at 12.20 (epoch 2800). Both then drift back up.",
         color=MUTED, fontsize=7.8, ha="left", va="top", linespacing=1.45)

fig.savefig(OUT, facecolor=SURFACE)
print(f"wrote {OUT}")
