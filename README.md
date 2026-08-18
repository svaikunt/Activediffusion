# Active Diffusion

Score-based generative models driven by **active matter** dynamics: instead of the
standard Ornstein–Uhlenbeck forward process, image pixels are coupled to a persistent
coloured-noise field $\eta$ with correlation time $\tau$.

```
passive (baseline)      dx = -k·x dt + √(2Tp) dW

active                  dx = (-k·x + η) dt + √(2Tp) dW_x
                        dη = (-η/τ)     dt + (√(2Ta)/τ) dW_η
```

The forward kernel is Gaussian in the joint variable $z=(x,\eta)$, so the model learns the
joint score $\nabla \log p_t(x,\eta)$ and samples by reverse-time integration of the
two-variable SDE (or its probability-flow ODE).

---

![Reverse-time active sampling](active_sampling.gif)

Reverse-time sampling from the τ=0.5 model. Both panels are on **fixed** scales, so the
amplitudes are real. The image `x` forms on the left; the active noise `η` on the right
never changes amplitude — it is an autonomous OU process already at equilibrium, so its
marginal is the same at every *t*.

What changes is the *relationship*. At t = T the two panels are near-copies of each other,
`corr(x, η) = 0.82` — the stationary correlation. As t → 0, `x` acquires image statistics
(nearest-neighbour pixel correlation rises from 0.00 to 0.93) and **decouples** from the
field that drove it, with the correlation falling to zero. The trace underneath tracks it.

![Noise to t*=0.2, then reverse](active_recovery.gif)

**Does the image come back?** A real image is noised to t\*=0.2 — visually pure noise, with
signal variance 0.05 against 0.245 of noise — and then reversed. Both arms start from the
*identical* noised state; the only difference is that the right column has η permuted
across the batch, which preserves η's distribution exactly and destroys only its pairing
with x.

With matched η the originals come back to **6.8%** of the pixel variance. With η shuffled,
**321%** — further from the original than two unrelated images would be (~200%), so a
mismatched η does not merely fail to help, it actively misleads. The same experiment at
t\*=0.10 and 0.15 gives 1.6% vs 103% and 3.7% vs 206%.

Note what this does and doesn't say: η₀ is drawn independently of x₀, so η carries no
information about the *image*. It carries information about the *noise that was added* —
which is what lets the reverse process subtract the contamination instead of guessing at
it. That is the mechanism behind the active model's advantage.

---

---

## Current result — CIFAR-10, unconditional

All rows below are the **same architecture, optimizer, EMA, batch size, learning-rate
schedule and training budget** — epoch 1000, batch 128, lr 2e-4 with 100k warmup. They
differ only in the SDE and, for the two τ=0.5 rows, only in the loss weighting.

| model | loss | FID |
|---|---|---|
| **active, τ=0.5** | **σ-whitened (same as passive)** | **4.98** |
| active, τ=0.5 | original (`√det M`) | 6.91 |
| active, τ=0.15 | original (`√det M`) | 9.50 |
| **passive baseline** | standard (`σ·score = −ε`) | **14.51** † |

Measured with `clean-fid` (`legacy_pytorch`) against the CIFAR-10 train split,
N = 50,000, quadratic probability-flow sampler at 500 steps, Heun solver.

The table is fixed at epoch 1000 so that all four rows are matched. The **best result
obtained is FID 4.88**, at epoch 1200, where the σ-whitened run was stopped:

| epoch | 800 | 1000 | 1200 |
|---|---|---|---|
| active τ=0.5, σ-whitened | 5.67 | 4.98 | **4.88** |
| ratio to previous | — | 0.878 | 0.980 |

The improvement rate collapsed from 0.878 to 0.980 in one interval, so the run was at or
near its floor; every other per-200-epoch ratio in this project falls in 0.90–0.97, and in
an earlier run a ratio of 0.972 immediately preceded the minimum and a subsequent rise.

**Active beats the passive baseline by 2.9× at matched training budget.** Most of that
comes from the SDE itself — 14.51 → 6.91 is a factor of 2.1 — with the loss fix contributing
a further 1.39× on top (6.91 → 4.98). See
[`CIFAR10/whitening_note.pdf`](CIFAR10/whitening_note.pdf).

† The checkpoint behind the 14.51 (`passive_cld_lr2e4`) was lost to a scratch purge, so
that number is a valid past measurement but cannot be re-scored. The strongest passive
result still backed by a surviving checkpoint is **13.96** (`passive_cld_lr1e4`, epoch
1600, N = 50,000) — against which the σ-whitened active model is **2.8× better on
40% less training**. Prefer that comparison when the difference matters.

Comparisons against published models are deliberately left out here. Every row above shares
one architecture, one optimizer and one budget, which is what makes the active-vs-passive
difference attributable to the SDE and the loss. Published CIFAR-10 numbers come from
different architectures, parameter counts, training lengths and — in the discrete-time
case — learned noise schedules, so setting them beside these would compare four things at
once. They are recorded in [`CIFAR10/FID_RESULTS.md`](CIFAR10/FID_RESULTS.md) where the
caveats can be stated properly.

---

## Layout

| Directory | Contents |
|---|---|
| [`CIFAR10/`](CIFAR10/) | Main experiments — training, sampling, FID, results log |
| [`MNIST/`](MNIST/) | Earlier single-channel prototype |
| [`Belief_Propagation/`](Belief_Propagation/) | Separate line of work |

Start with [`CIFAR10/README.md`](CIFAR10/README.md) for how to train and sample, and
[`CIFAR10/FID_RESULTS.md`](CIFAR10/FID_RESULTS.md) for the full experimental record —
every run, its configuration, and what is and isn't reproducible.

---

## Technical notes

Four standalone write-ups, each with LaTeX source alongside:

| Note | Subject |
|---|---|
| [`vanloan_note.pdf`](CIFAR10/vanloan_note.pdf) | Van Loan's method for the transition covariance: why the integral is a block of a matrix exponential, and why the hand-derived closed form failed |
| [`whitening_note.pdf`](CIFAR10/whitening_note.pdf) | The loss whitening: why a single `√det M` cannot whiten two channels, and the per-channel fix |
| [`loss_comparison_note.pdf`](CIFAR10/loss_comparison_note.pdf) | Active vs passive losses, comparison with CLD, the τ→0 limit |
| [`active_sscs_sampler_note.pdf`](CIFAR10/active_sscs_sampler_note.pdf) | The symmetric splitting (SSCS) sampler |
| [`tweedie_denoising_note.pdf`](CIFAR10/tweedie_denoising_note.pdf) | The final Tweedie denoising step |

---

## Two findings worth knowing before reusing this code

**1. The transition covariance must be computed by Van Loan's method, not in closed form.**
The covariance $M(t)=\int_0^t e^{As}Qe^{A^\top s}\,ds$ has a hand-derived closed form with
$(k-1/\tau)^2$ in the denominator. It divides by zero at $k\tau=1$ (where the drift matrix
is defective — a Jordan block) and loses all precision to cancellation at small $t$
elsewhere: measured relative error in $M_{11}$ at $t=10^{-3}$ reaches **241%** at τ=0.2,
which is enough to make the matrix non-positive-definite and crash the Cholesky. The
current implementation obtains $M$ as a block of a single 4×4 matrix exponential, which is
exact at every τ.

**2. The active loss needs per-channel whitening.** The passive loss is whitened —
`σ·score = −ε` has unit variance at every noise level. The active loss originally used the
scalar `√det M` for both channels, but `√det M = √v_x·√M₂₂ = √v_η·√M₁₁`, so it is the
correct whitener times a spurious factor. At τ=0.5 the η target collapsed by 515× between
t=T and t=10⁻³. Dividing each channel by its own σ — exactly what the passive loss already
does, just per channel because one scalar cannot serve two — restores unit-variance targets
and accounts for the 6.91 → 4.98 improvement.

Both changes are verified in [`CIFAR10/redteam_cond.py`](CIFAR10/redteam_cond.py) and
[`CIFAR10/redteam_cond_weights.py`](CIFAR10/redteam_cond_weights.py), which check the
shipped classes rather than a reimplementation.

---

## Reproducibility

Results predating the Van Loan patch (commit `1b25438`) were trained against a numerically
corrupted forward process and **cannot be reproduced from the current code**. They are
flagged individually in `FID_RESULTS.md`. Do not compare across that boundary: the error
varies per arm, being largest where $|k - 1/\tau|$ is smallest.
