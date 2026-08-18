"""Where does recovery stop working?  Noised state and recovery, three depths.

Aggregate MSE hides the failure mode: past t*~0.45 most tiles come back as a
*different* picture with the same composition and palette, which a single number
reports as "somewhat worse" rather than "wrong image".  Hence per-image
correlation, and hence showing the tiles.

python plot_recovery_limit.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 5                                   # 5x5 subset so individual tiles are legible
NG = G * G
TS = ["0.40", "0.45", "0.50"]
D = "DownloadsFromCluster/recover_t{}.npz"

o = np.load(D.format("0.30"))["orig"].astype(np.float32)[:NG]

def tile(b):
    b = b.reshape(G, G, 3, 32, 32).transpose(0, 3, 1, 4, 2)
    return np.clip((b.reshape(G * 32, G * 32, 3) + 1) / 2, 0, 1)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
AXIS, C_OK, C_BAD = "#c3c2b7", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": SURFACE})

fig = plt.figure(figsize=(11.2, 6.35), dpi=200)
W, Hh, X0, Y1, Y0 = 0.222, 0.352, 0.032, 0.455, 0.062

def panel(x, y, img, title, col, sub=None):
    ax = fig.add_axes([x, y, W, Hh])
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(AXIS); s.set_linewidth(1.0)
    ax.set_title(title, color=col, fontsize=10, fontweight="bold", pad=5)
    if sub:
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
                color=MUTED, fontsize=8)
    return ax

panel(X0, Y0, tile(o), "original", INK)
fig.text(X0 + W / 2, Y1 + Hh / 2, "noised state\nis shown above\neach recovery",
         color=MUTED, fontsize=9, ha="center", va="center", linespacing=1.6)

rows = []
for j, T in enumerate(TS):
    d = np.load(D.format(T))
    m = d["matched"].astype(np.float32)
    start, rec = m[0][:NG, [0, 2, 4]], m[-1][:NG, [0, 2, 4]]
    full = np.load(D.format(T))["matched"].astype(np.float32)[-1][:, [0, 2, 4]]
    ofull = np.load(D.format("0.30"))["orig"].astype(np.float32)
    c = np.array([np.corrcoef(full[i].ravel(), ofull[i].ravel())[0, 1]
                  for i in range(len(ofull))])
    mse = ((full - ofull) ** 2).mean() / ofull.var()
    x = X0 + (j + 1) * 0.242
    panel(x, Y1, tile(start), f"t* = {T}", INK)
    good = np.median(c) >= 0.75
    panel(x, Y0, tile(rec), f"recovered   {mse:.0%}",
          C_OK if good else C_BAD,
          f"median per-image corr {np.median(c):.2f} · {int((c<0.8).sum())}/49 below 0.8")
    rows.append((T, mse, np.median(c), int((c < 0.8).sum())))

fig.text(X0, 0.975, "Where recovery stops working",
         color=INK, fontsize=14, fontweight="bold", ha="left", va="top")
fig.text(X0, 0.930,
         "Real images noised to t*, then reversed with the η they were paired with. "
         "Same 25 images throughout; statistics over all 49.",
         color=MUTED, fontsize=8.6, ha="left", va="top")
fig.text(X0, 0.893,
         "By t*=0.45 the majority of tiles return as a different picture sharing the "
         "composition and palette — aggregate MSE alone does not show this.",
         color=MUTED, fontsize=8.6, ha="left", va="top")

fig.savefig("recovery_limit.png", facecolor=SURFACE)
print("wrote recovery_limit.png")
for r in rows:
    print(f"  t*={r[0]}  MSE {r[1]:.1%}  median corr {r[2]:.3f}  below0.8 {r[3]}/49")
