"""Training-loss curves for the active/passive CIFAR-10 runs.

Reads the sbatch .log files directly: the `ARGS:` banner gives score_param, tau,
Tp, Ta, k, T and batch size, and `Average Loss:` gives one point per epoch.

The point of the right-hand panel: the three objectives are NOT on a common
scale, so raw loss values cannot be compared across score_param.  With the
network output at zero the loss equals the mean squared target, and that
initialisation value is known in closed form:

    cond      Tp + Ta                       (targets are unit-variance at all t)
    whitened  2                             (r ~ N(0, I) at all t)
    score     Tp*E_t[M22] + Ta*E_t[M11]     (targets collapse as t -> 0)

Dividing by it gives the fraction of target variance still unexplained, which
IS comparable.  For tau=0.5, Ta=6.4, Tp=1e-3 the two are 6.401 and 2.503 -- so
a raw `cond` loss of 0.56 and a raw `score` loss of 0.068 are 8.7% and 2.7%
unexplained, a 3x gap rather than the 8x the raw numbers suggest.

    python plot_loss.py run1.log run2.log ...          # x = epoch
    python plot_loss.py --x step   *.log               # optimizer steps
    python plot_loss.py --x image  *.log               # images seen

Several logs sharing a run (resumes) can be passed together with --merge NAME.
"""

import argparse
import math
import os
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- log parsing
_ARG_RE = re.compile(r"--([A-Za-z0-9_]+)(?:\s+([^\s-][^\s]*))?")
_EPOCH_RE = re.compile(r"Epoch \[(\d+)/\d+\] Complete")
_LOSS_RE = re.compile(r"Average Loss:\s*([0-9.eE+-]+)")
_BATCHES_RE = re.compile(r"Dataset loaded:\s*(\d+) batches per GPU")
_WORLD_RE = re.compile(r"World Size:\s*(\d+)")


def parse_log(path):
    """-> dict(epochs, losses, args, steps_per_epoch, world, name)."""
    with open(path, errors="replace") as fh:
        text = fh.read()

    args = {}
    for line in text.splitlines():
        if line.startswith("ARGS:"):
            for key, val in _ARG_RE.findall(line[5:]):
                args[key] = val if val is not None else "true"
            break

    # `Average Loss:` always follows its `Epoch [n] Complete` banner
    epochs, losses, pending = [], [], None
    for line in text.splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            pending = int(m.group(1))
            continue
        m = _LOSS_RE.search(line)
        if m and pending is not None:
            epochs.append(pending)
            losses.append(float(m.group(1)))
            pending = None

    spe = _BATCHES_RE.search(text)
    world = _WORLD_RE.search(text)
    return dict(
        epochs=epochs,
        losses=losses,
        args=args,
        steps_per_epoch=int(spe.group(1)) if spe else None,
        world=int(world.group(1)) if world else 1,
        name=os.path.basename(path),
    )


def fget(args, key, default):
    try:
        return float(args[key])
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------- initial loss
def _expm(M, terms=80):
    """Scaling-and-squaring expm; pure numpy so this runs anywhere torch does."""
    n = max(0, int(np.ceil(np.log2(max(np.abs(M).max(), 1e-300)))) + 5)
    A = M / (2.0 ** n)
    E = np.eye(M.shape[0])
    Tm = np.eye(M.shape[0])
    for i in range(1, terms):
        Tm = Tm @ A / i
        E = E + Tm
    for _ in range(n):
        E = E @ E
    return E


def cov_vanloan(t, k, tau, Tp, Ta):
    """Conditional Cov[(x_t, eta_t) | x_0, eta_0] -- matches compute_covariance."""
    w = 1.0 / tau
    A = np.array([[-k, 1.0], [0.0, -w]])
    Q = np.diag([2 * Tp, 2 * Ta / tau ** 2])
    C = np.block([[-A, Q], [np.zeros((2, 2)), A.T]])
    E = _expm(C * t)
    S = E[2:, 2:].T @ E[:2, 2:]
    return 0.5 * (S + S.T)


def init_loss(args, n_grid=4001):
    """Loss at zero network output, i.e. the mean squared target, E over t ~ U(t_eps, T).

    Returns (value, formula_string).  None if the run is passive (no closed form
    needed here -- passive already whitens, its target is -eps and init is 1).
    """
    if "active" not in args:
        return 1.0, "1  (passive: sigma*score = -eps)"

    mode = args.get("score_param", "score")
    Tp = fget(args, "Tp", 1e-3)
    Ta = fget(args, "Ta", 6.4)
    tau = fget(args, "tau", 0.5)
    k = fget(args, "k", 4.0)
    T = fget(args, "T", 2.0)

    if mode == "cond":
        # v_x * E[s_x^2] = 1 and v_e * E[s_e^2] = 1 at every t, exactly.
        return Tp + Ta, f"Tp + Ta = {Tp + Ta:.4g}"
    if mode == "whitened":
        return 2.0, "2  (r ~ N(0, I))"

    # score: Var(Feta) = M11 and Var(Fx) = M22, averaged over t
    ts = np.linspace(1e-3, T, n_grid)
    m11 = np.empty(n_grid)
    m22 = np.empty(n_grid)
    for i, t in enumerate(ts):
        S = cov_vanloan(t, k, tau, Tp, Ta)
        m11[i], m22[i] = S[0, 0], S[1, 1]
    val = Ta * np.trapezoid(m11, ts) / (T - 1e-3) + Tp * np.trapezoid(m22, ts) / (T - 1e-3)
    return float(val), f"Ta*E[M11] + Tp*E[M22] = {val:.4g}"


# ---------------------------------------------------------------- palette
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9b5de5", "#00b4d8"]


def label_for(run):
    a = run["args"]
    if "active" in a:
        bits = [f"active · {a.get('score_param', 'score')}", f"τ={fget(a, 'tau', 0):g}"]
    else:
        bits = ["passive"]
    bits.append(f"batch {int(fget(a, 'batch_size', 0)) * run['world']}")
    return "  ·  ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--x", choices=["epoch", "step", "image"], default="epoch")
    ap.add_argument("--smooth", type=int, default=1, help="centred moving average, in epochs")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N epochs (warmup transient)")
    ap.add_argument("--out", default="loss_curves.png")
    args = ap.parse_args()

    runs = []
    for path in args.logs:
        r = parse_log(path)
        if not r["epochs"]:
            print(f"  skipped {r['name']} — no 'Average Loss:' lines")
            continue
        runs.append(r)
    if not runs:
        raise SystemExit("nothing to plot")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=200)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.775, bottom=0.155, wspace=0.22)

    xlabel = {"epoch": "training epoch",
              "step": "optimizer steps",
              "image": "images seen"}[args.x]

    for run, col in zip(runs, COLORS * 4):
        ep = np.array(run["epochs"], float)
        ls = np.array(run["losses"], float)
        order = np.argsort(ep, kind="stable")
        ep, ls = ep[order], ls[order]
        keep = ep > args.skip
        ep, ls = ep[keep], ls[keep]
        if ep.size == 0:
            continue

        if args.smooth > 1:
            kern = np.ones(args.smooth) / args.smooth
            ls = np.convolve(ls, kern, mode="valid")
            ep = ep[args.smooth // 2: args.smooth // 2 + ls.size]

        spe = run["steps_per_epoch"] or 1
        bpg = fget(run["args"], "batch_size", 128)
        x = {"epoch": ep,
             "step": ep * spe,
             "image": ep * spe * bpg * run["world"]}[args.x]

        L0, formula = init_loss(run["args"])
        lab = label_for(run)

        axL.semilogy(x, ls, color=col, lw=1.8, zorder=3, label=lab)
        axL.axhline(L0, color=col, lw=1, ls=(0, (1, 3)), alpha=0.6, zorder=1)
        axL.annotate(f"init {L0:.3g}", xy=(x[-1], L0), xytext=(-2, 4),
                     textcoords="offset points", color=col, fontsize=7.5,
                     ha="right", alpha=0.9)

        axR.semilogy(x, ls / L0, color=col, lw=1.8, zorder=3, label=lab)
        axR.annotate(f"{100 * ls[-1] / L0:.1f}%", xy=(x[-1], ls[-1] / L0),
                     xytext=(6, 0), textcoords="offset points", color=col,
                     fontsize=8.5, fontweight="bold", va="center")

        print(f"{run['name']}")
        print(f"    {lab}")
        print(f"    epochs {int(ep[0])}–{int(ep[-1])}   final loss {ls[-1]:.5f}")
        print(f"    init   {formula}")
        print(f"    unexplained fraction  {ls[-1] / L0:.4f}  ({100 * ls[-1] / L0:.2f}%)")

    for ax in (axL, axR):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(1.0)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
        ax.set_xlabel(xlabel, color=INK2, fontsize=10, labelpad=6)

    axL.set_ylabel("average loss per epoch", color=INK2, fontsize=10, labelpad=6)
    axR.set_ylabel("loss / loss at initialisation", color=INK2, fontsize=10, labelpad=6)
    axR.set_title("fraction of target variance unexplained — comparable across objectives",
                  color=INK2, fontsize=9, pad=8)
    axL.set_title("raw value — NOT comparable across objectives",
                  color=INK2, fontsize=9, pad=8)

    leg = axL.legend(loc="lower left", frameon=False, fontsize=8.5, labelspacing=0.45)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.text(0.065, 0.955, "Active diffusion — training loss", color=INK,
             fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.065, 0.905,
             "Each objective has a different loss at zero output, so raw curves sit at "
             "different heights for reasons that have nothing to do with model quality.",
             color=MUTED, fontsize=8.5, ha="left", va="top")
    fig.text(0.065, 0.868,
             "cond init = Tp + Ta (unit-variance targets at every t) · score init = "
             "Ta·E[M11] + Tp·E[M22] (targets collapse as t → 0). Right panel divides it out.",
             color=MUTED, fontsize=8.5, ha="left", va="top")

    fig.savefig(args.out, facecolor=SURFACE)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
