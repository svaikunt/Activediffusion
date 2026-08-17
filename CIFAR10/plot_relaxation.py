"""How fast does the joint (x, eta) distribution reach its stationary state?

Data: three-well x0 at {-1, 0, +1} (weights .4/.2/.4, width .05), eta0 drawn from its
equilibrium N(0, Ta/tau).  Covariance by Van Loan (exact at k*tau = 1), then marginalised
over eta0 and over the data.

Left  : x-eta correlation rho(t), dotted lines = stationary values.
Right : KL( joint(t) || stationary ), log scale.  Vertical line = the T=2 used in training.

python plot_relaxation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, math

# ---------------------------------------------------------------- model
Ta, Tp, k = 6.4, 1e-3, 4.0
TAUS = [0.15, 0.25, 0.50, 1.00]
T_TRAIN = 2.0

wells = np.array([-1.0, 0.0, 1.0]); wts = np.array([0.4, 0.2, 0.4]); sw = 0.05
VDATA = float((wts * wells**2).sum() + sw * sw)          # mean is 0 by symmetry


def _expm(M, terms=80):
    n = max(0, int(np.ceil(np.log2(max(np.abs(M).max(), 1e-300)))) + 5)
    A = M / (2.0**n)
    E = np.eye(M.shape[0]); Tm = np.eye(M.shape[0])
    for i in range(1, terms):
        Tm = Tm @ A / i; E = E + Tm
    for _ in range(n):
        E = E @ E
    return E


def cov_marginal(t, tau):
    """Cov of (x_t, eta_t) marginalised over eta0 ~ N(0, Ta/tau) and over the data."""
    w = 1.0 / tau
    A = np.array([[-k, 1.0], [0.0, -w]])
    Q = np.diag([2 * Tp, 2 * Ta / tau**2])
    C = np.block([[-A, Q], [np.zeros((2, 2)), A.T]])
    E = _expm(C * t)
    S = E[2:, 2:].T @ E[:2, 2:]
    S = 0.5 * (S + S.T)                                   # conditional (Van Loan)

    S0 = Ta / tau
    d = k - w
    a = math.exp(-k * t); c = math.exp(-t / tau)
    bm = (t * a) if abs(d) < 1e-12 else a * math.expm1(d * t) / d
    S = S.copy()
    S[0, 0] += bm * bm * S0 + a * a * VDATA
    S[0, 1] += bm * c * S0
    S[1, 0] = S[0, 1]
    S[1, 1] += c * c * S0
    return S


def cov_stationary(tau):
    see = Ta / tau
    sxe = see / (k + 1.0 / tau)
    return np.array([[sxe / k + Tp / k, sxe], [sxe, see]])


def kl(S, Sinf):
    return 0.5 * (np.trace(np.linalg.solve(Sinf, S)) - 2
                  + math.log(np.linalg.det(Sinf) / np.linalg.det(S)))


# ---------------------------------------------------------------- palette
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # validated slots 1-4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=200)
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.145, wspace=0.24)

ts = np.concatenate([np.linspace(1e-3, 0.3, 200), np.linspace(0.3, 3.0, 400)])

for tau, col in zip(TAUS, COLORS):
    Sinf = cov_stationary(tau)
    rinf = Sinf[0, 1] / math.sqrt(Sinf[0, 0] * Sinf[1, 1])
    rho, kls = [], []
    for t in ts:
        S = cov_marginal(t, tau)
        rho.append(S[0, 1] / math.sqrt(S[0, 0] * S[1, 1]))
        kls.append(max(kl(S, Sinf), 1e-18))               # clip for the log axis

    axL.plot(ts, rho, color=col, lw=2, zorder=3)
    axL.axhline(rinf, color=col, lw=1, ls=(0, (1, 3)), alpha=0.55, zorder=1)
    axL.annotate(f"τ={tau:g}", xy=(3.0, rho[-1]), xytext=(6, 0),
                 textcoords="offset points", color=col, fontsize=9.5,
                 fontweight="bold", va="center")

    axR.semilogy(ts, kls, color=col, lw=2, zorder=3)
    kl_at_T = max(kl(cov_marginal(T_TRAIN, tau), Sinf), 1e-18)
    axR.plot([T_TRAIN], [kl_at_T], "o", color=col, ms=7,
             mec=SURFACE, mew=2, zorder=4)
    axR.annotate(f"τ={tau:g}   {kl_at_T:.0e}", xy=(T_TRAIN, kl_at_T),
                 xytext=(10, -2), textcoords="offset points",
                 color=col, fontsize=8.5, fontweight="bold", va="center")

for ax in (axL, axR):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
    ax.axvline(T_TRAIN, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(0, 3.35)
    ax.set_xlabel("diffusion time  t", color=INK2, fontsize=10, labelpad=6)

axL.set_ylabel("corr(x, η)", color=INK2, fontsize=10, labelpad=6)
axL.set_ylim(0, 1.0)
axL.annotate("T = 2\n(training)", xy=(T_TRAIN, 0.06), xytext=(-8, 0),
             textcoords="offset points", color=MUTED, fontsize=8,
             ha="right", linespacing=1.4)

axR.set_ylabel("KL( joint(t) ‖ stationary )", color=INK2, fontsize=10, labelpad=6)
axR.set_ylim(1e-17, 1e2)

fig.text(0.065, 0.955, "Relaxation of the joint (x, η) distribution — CIFAR-scale "
         "parameters, three-well data",
         color=INK, fontsize=13.5, fontweight="bold", ha="left", va="top")
fig.text(0.065, 0.905,
         "k = 4 · Ta = 6.4 · Tp = 1e-3 · x₀ ∈ {−1, 0, +1} · η₀ from equilibrium.  "
         "Dotted = stationary correlation.  Covariance via Van Loan.",
         color=MUTED, fontsize=8.5, ha="left", va="top")
fig.text(0.065, 0.868,
         "Larger τ relaxes more slowly — four orders of magnitude across the sweep — but "
         "every arm is converged well before T = 2.",
         color=MUTED, fontsize=8.5, ha="left", va="top")

fig.savefig("relaxation_x_eta.png", facecolor=SURFACE)
print("wrote relaxation_x_eta.png")
