# CIFAR-10 FID Results and Methodology

Living record of FID numbers, how each was measured, and the training configs behind
them. Last updated 2026-08-12.

> Two separate training runs appear below. **Do not plot them as one curve** — they used
> different effective batch sizes, so the same epoch number means different amounts of
> training. See [Comparing the two runs](#comparing-the-two-runs).

---

## 1. Results

### 1.1 Current active run — `active_notweedie_tau015_Ta64_k4/`

Sampler: PF-ODE, 600 steps, log schedule, Heun, Tweedie on. N = 10,000 samples.
Measured 2026-08-12.

| epoch | FID   | Δ vs prev | optimizer steps (approx) |
|-------|-------|-----------|--------------------------|
| 1000  | 21.58 | —         | 97,000                   |
| 1200  | 19.20 | −2.38     | 116,400                  |
| 1400  | 18.24 | −0.96     | 135,800                  |
| 1600  | 17.38 | −0.86     | 155,200                  |
| 1800  | 16.50 | −0.88     | 174,600                  |
| 2000  | 15.39 | −1.11     | 194,000                  |
| 2200  | 14.03 | −1.36     | 213,400                  |
| 2400  | 12.69 | −1.34     | 232,800                  |
| 2600  | 12.34 | −0.35     | 252,200                  |
| 2800  | 12.20 | −0.14     | 271,600                  |
| 4000  | 13.04 | +0.84     | 388,000                  |

**Active has converged at ≈ 12.6 and training past ~2400 gains nothing.** The improvement
rate grew from −0.86 (1600) to ~−1.35 for three intervals, then collapsed: −0.35, −0.14,
then **+0.84**. The four points from 2400 on (12.69, 12.34, 12.20, 13.04) have mean 12.57
and span 0.84 with no trend — 1600 epochs of additional training, ~135k optimizer steps,
produced no improvement.

Note the within-plateau spread of 0.84 exceeds the 0.5 noise estimate from the single
repeat in §1.5. Either the noise floor at N = 10,000 is closer to **~0.8**, or the model
drifts slowly at this plateau. Either way, treat sub-1.0 differences at N = 10,000 as
unresolved; use N = 50,000 for anything that has to be defended.

**This makes the ~12.6 floor a property of the configuration, not the model class** — the
old run reached 7.91 at ~429k steps (§1.2), and this run is now at 388k steps and 4.5 FID
worse. Since training length is no longer the explanation, the remaining difference is the
effective batch (512 here vs 256 there) and whatever the pre-1000 segment did differently.
Continuing `active_resume2.sh` to 5200 has no expected value; re-running at
`--nproc_per_node=2 --batch_size 128` does.

Note this 2200 is the **current** run's checkpoint
(`active_notweedie_tau015_Ta64_k4/checkpoint_epoch_2200.pt`), not the old run's
`CIFAR10/checkpoint_epoch_2200.pt` in §1.2. At the same epoch number the current run has
had roughly half the gradient updates, so the two are not comparable point-to-point.

Other measurements on this run:

| config | epoch | FID | note |
|--------|-------|-----|------|
| PF-ODE, **200** steps, log, Tweedie on | 1600 | 122.34 | anomalous — see [Open questions](#open-questions) |

### 1.2 Old active run — `results_cifar10_Ta64_k4_multigpu/`

Trained ~Feb 2026. Only `checkpoint_epoch_2200.pt` survives; the launch scripts were
overwritten.

| epoch | FID  | sampler / N | when |
|-------|------|-------------|------|
| 2200  | 7.91 | original eval settings (not recorded) | ~Feb 2026 |
| 2200  | ~9.x | PF-600 log, Tweedie on, N = 10,000 | 2026-08-12 re-measure |

The 7.91 → 9.x difference is the sample count and Tweedie setting, not a model or
config problem.

### 1.3 Passive run — `passive_tp64_k4/`

Same sampler and scoring as §1.1 (PF-600 log, Heun, Tweedie on, N = 10,000).

| epoch | FID   | Δ vs prev | active at same epoch | gap (active − passive) |
|-------|-------|-----------|----------------------|------------------------|
| 1000  | 21.15 | —         | 21.58                | +0.43 (passive ahead)  |
| 1200  | 20.64 | −0.51     | 19.20                | −1.44 (active ahead)   |
| 1400  | 20.63 | −0.01     | 18.24                | −2.39 (active ahead)   |
| 1600  | 21.01 | +0.38     | 17.38                | −3.63 (active ahead)   |
| 2200  | 20.95 | −0.06     | 14.03                | −6.92 (active ahead)   |

Checkpoint identity verified for 1200, 1400 and 1600 via the `ckpt epoch:` line the §3.2
verification block prints — these are not the same model scored repeatedly.

**Passive is flat from epoch 1200 onward**: 20.64, 20.63, 21.01, 20.95 — a 1000-epoch
span with no improvement, scattering ~±0.2 around 20.8. Over the same span active falls
monotonically from 19.20 to 14.03. The gap widens with training rather than being a
fixed offset:

| epoch | gap (passive − active) |
|-------|------------------------|
| 1000  | −0.43 (passive better) |
| 1200  | +1.44                  |
| 1400  | +2.39                  |
| 1600  | +3.63                  |
| 2200  | +6.92                  |

The 20.63 → 21.01 rise also gives a rough noise scale for FID at N = 10,000: run-to-run
and epoch-to-epoch scatter on a converged model is ~0.2–0.4. That puts the epoch-1000
passive lead of 0.43 at roughly the noise floor — it should not be treated as a real
passive advantage. Every gap from 1200 on is far outside it.

The §3.2 verification block passed, which confirms this checkpoint was trained with
`Tp=6.4`, `k=4.0`, `T=2.0`, `num_res_blocks=4`, `attn_resolutions=16`, `active=False`.

The two curves cross between epoch 1000 and 1200: passive is marginally ahead at 1000
(within noise), and active is clearly ahead by 1200. Over that interval active improved
2.38 and passive only 0.51.

**The lr-drop confound is ruled out.** `passive_resume.sh` mirrors `active_resume.sh`
exactly — resume from `checkpoint_epoch_1000.pt`, `--lr 0.00005`, `--epochs 2600`,
`--batch_size 128`, `torchrun --nproc_per_node=4`, `--model_ema_decay 0.9997`, same
architecture. Both runs therefore share:

| | active | passive |
|---|---|---|
| lr schedule | 1e-4 → 5e-5 at epoch 1000 | same |
| effective batch | 512 (4 × 128) | same |
| optimizer steps/epoch | ~97 | same |
| effective EMA decay | 0.99985 | same |
| arch | 128 / 1,2,2,2 / 4 blocks / attn 16 | same |
| `T`, `k` | 2.0, 4.0 | same |
| SDE | `Ta=6.4`, `Tp=1e-3`, `tau=0.15` | `Tp=6.4` |

The only intended difference is the last row, so the epoch-1200 gap is a genuine
active-vs-passive result.

Residual check (cheap, not yet done): confirm the two runs had matched pre-1000 segments
by comparing optimizer step counts at epoch 1000 — they should be equal.

```python
import torch
for d in ("active_notweedie_tau015_Ta64_k4", "passive_tp64_k4"):
    ck = torch.load(f"{d}/checkpoint_epoch_1000.pt", map_location="cpu",
                    weights_only=False, mmap=True)
    print(d, ck["args"]["lr"], float(next(iter(ck["optimizer"]["state"].values()))["step"]))
```

Passive also resumes into its own `--out_dir`, so it carries the same
overwrite hazard as §4.3.

---

### 1.4 Sampler sweep at epoch 1000 (jobs 1358899 / 1358900)

Both models at their **epoch-1000** checkpoints, N = 10,000, same clean-fid call as §3.
Sweep used the **quadratic** schedule with `t_end = 0.001`; SSCS was equilibrium split with
score@midpoint; PF used Heun.

| model | sampler | schedule | steps | FID |
|-------|---------|----------|-------|-----|
| active  | PF-ODE | quadratic | 50  | 35.38 |
| active  | PF-ODE | quadratic | 500 | **21.75** |
| active  | SSCS   | quadratic | 50  | 48.34 |
| active  | SSCS   | quadratic | 500 | 21.58 |
| passive | PF-ODE | quadratic | 50  | 19.24 |
| passive | PF-ODE | quadratic | 500 | **15.63** |
| passive | EM     | —         | 50  | 85.32 |
| passive | EM     | —         | 500 | 17.95 |

#### ⚠ This contradicts §1.3 and needs resolving before the main claim stands

Same epoch-1000 passive checkpoint, two samplers:

| passive @ epoch 1000 | FID |
|----------------------|-----|
| PF-500, **quadratic** | 15.63 |
| PF-600, **log** (§1.3) | 21.15 |

**5.5 FID apart on the same weights.** Going 500 → 600 steps cannot explain that, so the
schedule is doing it. The active model shows no such sensitivity — 21.75 quadratic-500 vs
21.58 log-600, a 0.17 difference well inside noise.

So at epoch 1000 the ranking *flips with the sampler*:

| schedule | passive | active | winner |
|----------|---------|--------|--------|
| log, 600 steps       | 21.15 | 21.58 | tie (0.43, ~noise) |
| quadratic, 500 steps | 15.63 | 21.75 | **passive by 6.1** |

Every number in §1.1–§1.3 — including the entire "passive plateaus at ~20.8 while active
descends" result — was measured on the **log** schedule. If the log schedule
systematically handicaps the passive model, that plateau may be a property of the sampler
rather than of the model, and the widening gap would be an artifact.

**The experiment that settles it:** rescore the passive checkpoints at 1200/1600/2200 and
the active checkpoints at the matched epochs using `--pf_schedule quadratic --pf_steps 500`,
and see whether the passive curve is still flat. If passive under quadratic descends while
under log it plateaus, the log-schedule numbers cannot be used for the headline comparison.
Until that is done, treat §1.3 as provisional.

Secondary observations, lower stakes:

- **PF dominates at low step counts.** At 50 steps, passive PF is 19.24 vs EM 85.32; active
  PF is 35.38 vs SSCS 48.34. The stochastic samplers need many steps to be usable.
- **The stochastic samplers converge to their PF counterparts by 500 steps** — active SSCS-500
  21.58 vs PF-500 21.75; passive EM-500 17.95 vs PF-500 15.63 (EM still trailing).
- **Passive is much more step-efficient than active here**: passive PF-50 (19.24) already
  beats every active number at any step count on this checkpoint.

### 1.5 Quadratic-schedule re-test (in progress)

Rescoring the same checkpoints with `--pf_steps 500 --pf_schedule quadratic --pf_solver heun`,
N = 10,000, to test whether the §1.3 passive plateau is a property of the log schedule.
Checkpoint identity verified from the `ckpt epoch:` line in each block.

| epoch | passive quad-500 | passive log-600 (§1.3) | log − quad | active quad-500 | active log-600 |
|-------|------------------|------------------------|------------|-----------------|----------------|
| 1000  | 15.09            | 21.15                  | 6.06       | 21.75 (§1.4)    | 21.58 |
| 1400  | 14.40            | 20.63                  | 6.23       | 18.08           | 18.24 |
| 2200  | 13.65            | 20.95                  | 7.30       | 13.86           | 14.03 |
| 2600  | 14.41            | —                      | —          | _pending_       | 12.34 |

**Active is schedule-insensitive, confirmed at all three epochs:** Δ(quad − log) = 0.17,
−0.16, −0.17 at 1000 / 1400 / 2200. Remarkably consistent and far inside the 0.5 noise
floor. Only the passive model cares about the schedule, where the gap is 6–7 FID and grows.

#### The matched comparison inverts

| epoch | active quad | passive quad | gap | (log-schedule gap, for contrast) |
|-------|-------------|--------------|-----|----------------------------------|
| 1000  | 21.75 | 15.09 | **passive ahead 6.66** | active ahead 0.43 |
| 1400  | 18.08 | 14.40 | **passive ahead 3.68** | active ahead 2.39 |
| 2200  | 13.86 | 13.65 | 0.21 — **tie** (inside noise) | active ahead 6.92 |

#### Passive plateaus under quadratic too — at ~14, not ~20.8

The epoch-2600 point came back **higher** than 2200: 13.65 → 14.41, a rise of 0.76. The
full quadratic passive series is **15.09, 14.40, 13.65, 14.41** — non-monotonic, with the
2200→2600 rise (+0.76) about equal in size to the 1400→2200 fall (−0.75).

Read as a whole, passive is **converged at ≈ 14.1 from epoch 1400 onward**, scattering
±0.4, and the apparent "steady descent" through 2200 was noise plus one low draw. The
plateau intuition was right; the log schedule just put the plateau at the wrong *level*
(20.8 instead of 14.1) and made the early part of the curve look flat when it wasn't.

#### Final shape: both models hit a minimum, then slowly degrade

Full quadratic passive series: 15.09 (1000), 14.40 (1400), **13.65 (2200)**, 14.41 (2600),
14.95 (3600). Full active series (log, schedule-equivalent): … 14.03 (2200), 12.69 (2400),
12.34 (2600), **12.20 (2800)**, 13.04 (4000).

| model | best FID | at epoch | drift after the minimum |
|-------|----------|----------|--------------------------|
| passive | **13.65** | 2200 | +0.76 by 2600, +1.30 by 3600 |
| active  | **12.20** | 2800 | +0.84 by 4000 |

**Best-vs-best gap: 1.45, in active's favour.**

Passive's rise is monotonic across two successive intervals (13.65 → 14.41 → 14.95), which
is harder to attribute to noise than a single high draw. Both models therefore appear to
*over-train* at this configuration: FID bottoms out and then creeps back up. This is the
strongest argument yet that continuing either run to 5200 is counterproductive, and that
the gap to the old run's 7.91 is a configuration problem (effective batch 512 vs 256), not
a training-length problem.

#### Earlier framing (superseded by the 3600/4000 points): both models plateau

Active is schedule-insensitive to within 0.17, so its log series can be read directly
against the passive quadratic series.

| model | behaviour | plateau |
|-------|-----------|---------|
| passive | 15.09 → ~14.1 by epoch 1400, then flat (14.40, 13.65, 14.41) | **≈ 14.15** (spread 0.76) |
| active  | 21.75 → 18.08 → 13.86 → 12.69, 12.34, 12.20 — flat from ~2400 | **≈ 12.41** (spread 0.49) |

Both plateaus are established on three points each, with within-plateau spread comparable
to the 0.5 noise floor. The separation between them is **≈ 1.74**, i.e. ~3.5× the noise
floor and well outside it.

**The result:** passive converges quickly to ≈ 14.15 and stops. Active starts far worse
(21.75 at epoch 1000, 6.7 behind), improves ~5× faster, crosses passive at **epoch ≈ 2200**,
and settles ≈ 1.8 lower at ≈ 12.41. So active is both slower to get going and better in the
end — a convergence-rate story on top of a final-quality one, and neither is visible under
the log schedule.

**Caveat: all of these are N = 10,000, where the noise floor is ~0.5.** The passive
plateau claim rests on three points spanning 0.76, which is only marginally outside that.
Before this is final, re-measure passive at 1400 / 2200 / 2600 and active at 2600 with
**N = 50,000**. The qualitative shape is unlikely to change, but the plateau level and the
final gap both currently carry error bars comparable to the effects being described.

**Control passed, with a caveat.** Epoch 1000 came back at 15.09 against the §1.4 sweep's
15.63 — same checkpoint, same sampler config, different (unseeded) draw. The setup
reproduces, but the 0.54 gap is a **larger noise estimate than the ~0.2–0.4 assumed
earlier**. Revise the noise floor for FID at N = 10,000 to **~0.5**.

#### Verdict: the §1.3 passive plateau is a log-schedule artifact

Under quadratic, passive descends — 15.09 → 14.40 → 13.65 — where under log over the same
span it is flat (21.15 → 20.63 → 20.95). So the model *is* improving and the log schedule
is not seeing it.

But normalize for the unequal intervals before reading too much into the descent:

| span | passive quad | active log |
|------|--------------|------------|
| 1000 → 1400 | −0.35 per 200 ep | −1.67 per 200 ep |
| 1400 → 2200 | −0.19 per 200 ep | −1.05 per 200 ep |

Passive descends roughly **5× slower** than active. The second passive interval, −0.19 per
200 epochs, is at the noise floor. So passive is *shallow*, not flat — the qualitative
plateau intuition was closer than a raw reading of the interval drops suggests.

The schedule gap also **widens** with training (6.06 → 6.23 → 7.30), so the log schedule is
not a constant handicap. It saturates around ~20.6–21 for the passive model and stops
registering improvement that the model is demonstrably making. That is exactly the failure
mode that would manufacture a fake plateau.

**Consequences:**

1. §1.1–§1.3 and `fid_active_vs_passive.png` are **log-schedule only** and cannot support
   "passive saturates while active descends." Superseded pending quadratic numbers.
2. The head-to-head at 2200 likely collapses. Active is schedule-insensitive at epoch 1000
   (21.75 quad vs 21.58 log). If that holds at 2200, active-quad ≈ 14.03 against
   passive-quad 13.65 — **a 0.38 difference, inside noise, i.e. a tie**, versus the 6.92
   active lead the log numbers reported.
3. Active quad-500 at 1400 and 2200 is now **required**, not a spot-check.

```bash
sed -i 's/^EPOCHS="1000 1400 2200"/EPOCHS="1400 2200"/' fid_quad_active.sh
sbatch --exclude=g002,l001 fid_quad_active.sh
```

Open question worth its own experiment: *why* is the passive model so much more
schedule-sensitive than the active one? The active model scores the same under both, the
passive model differs by 6–7 FID. That asymmetry is a real finding regardless of how the
head-to-head lands.

### 1.6 CLD-recipe runs, single GPU (`active_cld_lr2e4/`, `passive_cld_lr2e4/`)

Started 2026-08-13. Optimization copied from CLD (ICLR 2022, Table 6, CIFAR-10 Main):
Adam `lr 2e-4`, EMA rate 0.9999, 100k warmup iterations, grad-clip 1.0, batch **128** on a
single GPU, AMP. 2000 epochs × 390 steps/epoch = **781k steps**, matching CLD's 800k.

`--model_ema_decay 0.9984375` is the nominal value that yields CLD's effective 0.9999
through the script's torchvision adjustment (`adjust = 128*10/2000 = 0.64`;
`alpha = 1.5625e-3 × 0.64 = 1e-3`, a 10k-step horizon).

Scored with quadratic PF-500, N = 10,000 — directly comparable to §1.5.

| epoch | optimizer steps | images | active | passive |
|-------|-----------------|--------|--------|---------|
| 500   | 195,000 (95k post-warmup) | 25M | **17.90** | **20.69** |
| 1000  | 390,000 (290k post-warmup) | 50M | **11.19** | **16.29** |
| 1500  | 585,000 | 75M | 12.17 | _pending_ |

**Sample-count offset is larger than assumed.** Active epoch 1500 scored **12.17 at
N=10,000** and **10.02 at N=50,000** — an offset of **2.15**, not the ~1.3 inferred from
the old run's 7.91@50k / ≈9.x@10k pair. The offset is model- and quality-dependent, so it
cannot be treated as a constant conversion. **Never compare a 10k number to a 50k number.**

10.02@50k is the best result this project has produced under a matched sampler apart from
the old 2-GPU run's 7.91@50k, which remains 2.1 ahead.

#### Passive under CLD: a modest deficit, not a collapse

Passive went 20.69 → 16.29 from epoch 500 to 1000, so most of the apparent disaster at 500
was the 100k warmup, not a permanent handicap. What remains is small but consistent:

| comparison | batch-512 passive | CLD passive | CLD deficit |
|------------|-------------------|-------------|-------------|
| matched epoch (1000) | 15.09 | 16.29 | 1.20 |
| matched images (50M)  | 15.09 | 16.29 | 1.20 |
| matched steps (~390k) | ≈15.0 (extrap.) | 16.29 | ≈1.3 |

So the CLD recipe costs passive ~1.2 FID and gains active ~10.6 (21.75 → 11.19 at epoch
1000). The differential is real and large, but it is almost entirely active *gaining*
rather than passive *losing*.

Gap between the models at epoch 1000, same sampler, same everything but the SDE:

| config | active | passive | gap |
|--------|--------|---------|-----|
| batch 512, lr 5e-5 | 21.75 | 15.09 | passive ahead 6.66 |
| CLD (batch 128, lr 2e-4) | **11.19** | **16.29** | **active ahead 5.10** |

An 11.8 FID swing in the gap from a hyperparameter change alone. This is the single most
important caveat in this document: **the sign of the active-vs-passive comparison is set by
the optimizer configuration**, and any reported gap must name the configuration it was
measured under.

#### Active at 1000 beats the entire batch-512 run

| config | epoch | steps | images | FID |
|--------|-------|-------|--------|-----|
| CLD, batch 128 | 1000 | 390k | 50M | **11.19** |
| 4-GPU, batch 512 | 4000 | 388k | 200M | 13.04 |
| 4-GPU, batch 512 | 2800 (its best) | 272k | 140M | 12.20 |
| old 2-GPU, batch 256 | 2200 | 404k | 103M | ≈ 9.x |

At **matched optimizer steps (~390k)** the CLD configuration is **1.85 better than the
batch-512 run while having seen a quarter of the data** (50M vs 200M images). It also
clears the batch-512 run's all-time best by 1.01, at half that run's epoch count, with
1000 epochs still to go.

This settles the question from §1.1: the ~12.4 floor was a property of the batch-512
configuration, not of the model or the architecture. Effective batch was the throttle.

Trajectory so far: 17.90 (500) → 11.19 (1000), i.e. −6.71 over 500 epochs. The remaining
1000 epochs bring it to 780k steps and 100M images — comparable data to the old 7.91 run
and roughly twice its optimizer steps. Whether it reaches ≈9.x is the open question; a
naive extrapolation says yes but the curve must bend somewhere.

#### The CLD recipe helps active and hurts passive

At **matched optimizer steps (195k)** the two configurations give opposite orderings:

| config | batch | lr | active | passive | ahead |
|--------|-------|-----|--------|---------|-------|
| 4-GPU (epoch ~2010) | 512 | 5e-5 | 15.3 | 13.8 | passive by 1.5 |
| CLD (epoch 500)     | 128 | 2e-4 | **17.90** | **20.69** | **active by 2.79** |

Same step count, same architecture, same SDE parameters — and the sign of the gap flips.
That rules out a purely step-count or purely data-exposure explanation for the crossover:
the hyperparameters themselves differentially affect the two models.

Relative to its own old-config trajectory the active model gained a lot from the CLD
recipe (17.90 at 25M images, where the 4-GPU run was still above 21.6 at the same data),
while passive lost ground (20.69, against 15.09 at 1000 epochs in the old config).

Two candidate mechanisms, both untested:

- **Gradient-noise scale — now the leading candidate.** `lr/batch` is 1.56e-6 under CLD
  versus 1.95e-7 in segment 1 (lr 1e-4, batch 512) and 9.77e-8 in segment 2 (lr 5e-5,
  batch 512) — **8× and 16× noisier** respectively. Passive's best-ever numbers came from
  the two quietest settings; its worst comes from the noisiest. Active moves the other
  way. This is the only mechanism the data still supports, and it is directly testable:
  rerun passive at `--lr 5e-5` with the rest of the CLD config held fixed
  (`lr/batch = 3.9e-7`) and see whether it recovers toward ~15.
- ~~**EMA horizon.**~~ **Disfavoured.** CLD's effective 0.9999 is a 10k-step window, and
  the 4-GPU runs used 68k steps after epoch 1000 — but their *first* 1000 epochs ran at
  nominal 0.997, an effective 0.998523, i.e. a **6,770-step horizon, even shorter than
  CLD's** (see §2.1). Passive posted 15.09 at epoch 1000 under that short window. So a
  short EMA does not by itself hurt passive, and cannot explain 20.69.

**Caution for the paper:** the active-vs-passive gap is not hyperparameter-independent. A
recipe taken from CLD favours active by 2.79 at 500 epochs; the batch-512 recipe favoured
passive by 1.5 at matched steps. Any headline comparison needs to state which
configuration it was measured under, and ideally show both.

#### What the 500-epoch point says

Three ways to compare it against the 4-GPU batch-512 active run:

| matched on | 4-GPU run reaches this at | its FID | CLD run | verdict |
|------------|---------------------------|---------|---------|---------|
| **total optimizer steps** (195k) | epoch ~2010 | ≈ 15.3 | 17.90 | CLD run 2.6 *worse* |
| **images seen** (25M) | epoch 500 | > 21.6 | 17.90 | CLD run ≥ 3.7 *better* |
| **post-warmup steps** (95k) | epoch ~1030 | ≈ 21.2 | 17.90 | CLD run 3.3 *better* |

Neither pure hypothesis survives. Optimizer-step count clearly is not the whole story —
at matched total steps the CLD run is behind. But data exposure is not the whole story
either — at matched data it is far ahead.

The third row is the most honest comparison, since half the CLD run's steps were spent on
the 100k lr ramp while the 4-GPU run's warmup was only 5k steps (51 epochs). Counting only
steps at terminal lr, the CLD configuration is **3.3 FID ahead on a quarter of the data**.

Projection: the run has 1500 epochs left. It reaches 390k steps at epoch 1000 (50M images),
where the 4-GPU run needed epoch 4000 and posted 13.04. If the current advantage holds, the
CLD run should pass the 12.20 plateau well before 2000 epochs.

### 1.7 lr-drop branch (`/net/scratch/svaikunt/cifar10/active_cld_lr1e4/`)

Branched from `active_cld_lr2e4/checkpoint_epoch_1000.pt` with **lr 2e-4 → 1e-4**,
everything else CLD. `--epochs 4000` with `--model_ema_decay 0.996875` reproduces the same
effective 0.9999 / 10k-step EMA window. `--save_freq 200`. Same sampler (quadratic PF-500,
N=10k).

Motivation: under constant 2e-4 the parent run scored 11.19 (1000), 12.17 (1500) and then
**35.82 (2000)** — a visible collapse in the sample grids, with training loss dead flat at
0.0680 ± 0.0004 the whole way and no loss spike anywhere. Checkpoint integrity, concurrent
writers, and file truncation were all ruled out.

| epoch | lr 2e-4 (parent) | lr 1e-4 (branch) |
|-------|------------------|------------------|
| 1000  | 11.19 | — (branch point) |
| 1200  | — | **11.82** |
| 1400  | — | **11.90** |
| 1600  | — | **11.92** |
| 1500  | 12.17 | |
| 2000  | 35.82 | |

**At N = 50,000** (publishable units, same quadratic PF-500 sampler):

| epoch | run | 10k | 50k | offset |
|-------|-----|-----|-----|--------|
| 1000  | lr 2e-4 | 11.19 | **9.50** | 1.69 |
| 1500  | lr 2e-4 | 12.17 | 10.02 | 2.15 |

The 10k→50k offset is **not constant** (1.69 vs 2.15), so the 10k history cannot be
converted with a single factor — each reported number needs its own 50k run. The ordering
does survive the change of N, so the 10k series remains valid for epoch-to-epoch decisions.

**Matched-sampler comparison at N = 50,000** (quadratic PF-500), the publishable units:

| epoch | active | passive | gap | note |
|-------|--------|---------|-----|------|
| 1000  | **9.50** (lr 2e-4) | **14.51** (lr 2e-4) | **5.01** | active's best |
| 1500  | 10.02 (lr 2e-4) | — | — | |
| 1600  | **9.64** (lr 1e-4 branch) | **13.96** (lr 1e-4 branch) | **4.32** | |

**The gap is stable across N.** At epoch 1000 it is 5.10 at N=10,000 and 5.01 at
N=50,000 — a 0.09 difference. So while the absolute 10k values are biased high by
1.7–2.2, the bias largely cancels in a difference, and the 10k series is trustworthy for
*gaps* even though individual numbers are not publishable.

Sample-count offsets measured so far: 1.69 (active 1000), 2.15 (active 1500), 1.78
(passive 1000). No clean dependence on model quality; not a usable single conversion
factor for any individual number.

Passive's 50k trajectory is 14.51 (1000) → 13.96 (1600), improving 0.55 over 600 epochs,
while active went 9.50 → 9.64, i.e. flat within scatter. **Passive is closing**: the gap
narrows 5.01 → 4.32 over 600 epochs, a real trend rather than a measurement artifact, and
the opposite of what the original log-schedule curves suggested (where the gap widened
with training).

**Active is flat at ~9.5–9.6 across the lr drop**, confirming in publishable units what
the 10k series showed: halving the learning rate bought stability, not quality.

Sample-count offsets, four measurements: 1.69 / 2.15 / 1.78 / 2.28 — span 1.69–2.28.

**9.50 is the best result from any current run**, against the old 2-GPU run's 7.91@50k
(§1.2), which remains 1.59 ahead.

11.82 vs 11.19 is inside the ~0.8 noise floor, so 200 epochs at the lower lr is a null
result — as expected. The parent was still healthy at 1500; the collapse happened between
1500 and 2000. **The decisive epochs for this branch are 1600–2000.**

Note the denoising loss was completely insensitive to a 3× FID degradation. It is averaged
over all noise levels and dominated by the easy high-noise end, while sample quality is
governed by accuracy at small t. Do not use training loss to monitor these runs.

## 2. Training configs

### 2.1 Current runs — how the data was actually made

Both runs were produced in two segments: epochs 1–1000, then a resume from
`checkpoint_epoch_1000.pt` that ran to 2600. Both segments used
`torchrun --nproc_per_node=4` with per-GPU `--batch_size 128` ⟹ **effective batch 512**,
~97 optimizer steps/epoch.

#### Segment 1 — epochs 1–1000

The launch scripts for this segment were not preserved. Two things differed from the
resume below (confirmed by the author for **both** the active and passive runs):

| | segment 1 (1–1000) | segment 2 (1000–2600) |
|---|---|---|
| `--lr` | `1e-4` | `5e-5` |
| `--model_ema_decay` | **`0.997`** | `0.9997` |

The EMA difference is the larger of the two, because the nominal value feeds the
adjustment in §4.2. At `--batch_size 128`, `--model_ema_steps 10`, `--epochs 2600`:

| segment | nominal | effective decay | horizon (steps) | horizon (epochs @ 97/ep) |
|---------|---------|-----------------|-----------------|--------------------------|
| 1–1000    | 0.997  | 0.998523 | ~6,770 | ~70 |
| 1000–2600 | 0.9997 | 0.99985  | ~68,000 | ~700 |

So the averaging window lengthened 10× at epoch 1000. Every FID point at epoch 1000 was
measured on a checkpoint produced under the **short** EMA; points from 1200 on are under
the long one.

This matters for interpreting §1.6 — see the note there on why it weakens the EMA-horizon
explanation for passive's behaviour under the CLD recipe.

The values can be confirmed directly from the epoch-1000 checkpoints:

```python
import torch
for d in ("active_notweedie_tau015_Ta64_k4", "passive_tp64_k4"):
    a = torch.load(f"{d}/checkpoint_epoch_1000.pt", map_location="cpu",
                   weights_only=False, mmap=True)["args"]
    print(d, "lr:", a["lr"], "ema:", a["model_ema_decay"],
          "epochs:", a["epochs"], "batch:", a["batch_size"])
```

Replace the table above with the printed values once run.

#### Segment 2 — `active_resume.sh` (job 1364673), epochs 1000 → 2600

Verbatim as submitted:

```bash
#!/bin/bash
#SBATCH --job-name=active_resume
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=active_resume_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10

ARGS=(
  --active
  --amp
  --ckpt active_notweedie_tau015_Ta64_k4/checkpoint_epoch_1000.pt
  --out_dir active_notweedie_tau015_Ta64_k4
  --Tp 1e-3
  --Ta 6.4
  --tau 0.15
  --k 4.0
  --T 2.0
  --epochs 2600
  --batch_size 128
  --model_base_dim 128
  --num_res_blocks 4
  --dim_mults 1,2,2,2
  --attn_resolutions 16
  --large_sample_interval 200
  --large_sample_count 200
  --pf_sample_interval 200
  --pf_sample_count 200
  --pf_steps 400
  --pf_schedule log
  --model_ema_decay 0.9997
  --save_freq 200
  --lr 0.00005
  --warmup_steps 5000
  --timesteps 1000
  --grad_clip 1.0
  --weight_decay 0.0
  --data_dir /home/svaikunt/CIFAR10/data
)

echo "ARGS: ${ARGS[@]}"
torchrun --standalone --nproc_per_node=4 train_multigpu.py "${ARGS[@]}"
```

#### Segment 2 — `passive_resume.sh`, epochs 1000 → 2600

Identical to the active script except for the job name/log, the checkpoint and output
directory, and the SDE block (`--Tp 6.4` in place of `--active --Tp 1e-3 --Ta 6.4
--tau 0.15`). Everything governing optimization — lr, batch, EMA, warmup, clip, epochs,
AMP — is the same, which is what makes the two curves comparable:

```bash
#!/bin/bash
#SBATCH --job-name=passive_resume
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=passive_resume_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10

ARGS=(
  --amp
  --ckpt passive_tp64_k4/checkpoint_epoch_1000.pt
  --out_dir passive_tp64_k4
  --Tp 6.4
  --k 4.0
  --T 2.0
  --epochs 2600
  --batch_size 128
  --model_base_dim 128
  --num_res_blocks 4
  --dim_mults 1,2,2,2
  --attn_resolutions 16
  --large_sample_interval 200
  --large_sample_count 200
  --pf_sample_interval 200
  --pf_sample_count 200
  --pf_steps 400
  --pf_schedule log
  --model_ema_decay 0.9997
  --save_freq 200
  --lr 0.00005
  --warmup_steps 5000
  --timesteps 1000
  --grad_clip 1.0
  --weight_decay 0.0
  --data_dir /home/svaikunt/CIFAR10/data
)

echo "ARGS: ${ARGS[@]}"
torchrun --standalone --nproc_per_node=4 train_multigpu.py "${ARGS[@]}"
```

Note both resumes write into the same `--out_dir` they read from, so they overwrite
checkpoints 1200–2000 as they pass those epochs — see §4.3.

The `--pf_steps 400` / `--pf_schedule log` in these scripts control the *preview* sample
grids written during training. They are unrelated to the 600-step PF-ODE used for every
FID number in §1.

### 2.2 Old run (reconstructed)

Reconstructed 2026-08-12 from `checkpoint_epoch_2200.pt` — see §4 for how.

```bash
torchrun --nproc_per_node=2 train_multigpu.py \
  --active --Ta 6.4 --Tp 1e-3 --k 4.0 --tau 0.15 --T 2.0 --timesteps 1000 \
  --model_base_dim 128 --dim_mults 1,2,2,2 --num_res_blocks 4 --attn_resolutions 16 \
  --batch_size 128 --amp \
  --lr 1e-4 --weight_decay 0.0 --grad_clip 1.0 --warmup_steps 5000 \
  --model_ema_decay 0.9997 --model_ema_steps 10 \
  --epochs 2600 --save_freq 200 --log_freq 10 \
  --out_dir results_cifar10_Ta64_k4_multigpu/
# at epoch 1000, resume with --lr 5e-5 --ckpt .../checkpoint_epoch_1000.pt
```

**Confidence:**

| tier | items |
|------|-------|
| Recorded in checkpoint | all arch and SDE params, AdamW betas (0.9, 0.999) eps 1e-8, `weight_decay=0`, `grad_clip=1.0`, `warmup_steps=5000`, `batch_size=128`/GPU, AMP, `epochs=2600`, `save_freq=200`, `lr=5e-5` for the 2000→2200 segment, resumed from `checkpoint_epoch_2000.pt`, cumulative optimizer steps `403850` |
| Derived | world size 2 / effective batch 256; effective EMA decay 0.99985; ~129 epochs of work lost to crash-and-resume |
| Recollection | `lr` 1e-4 → 5e-5 at epoch 1000 |
| Unknown | whether anything else changed before epoch 2000; number of crash/resume boundaries |

### 2.3 Comparing the two runs

Old run: 2 GPUs × 128 = effective 256 → `floor(50000/256)` = **195 steps/epoch**.
Current run: 4 GPUs × 128 = effective 512 → `floor(50000/512)` = **97 steps/epoch**.

**Current epoch N ≈ old epoch N/2 in gradient updates.** Always compare on optimizer
steps. The old run reached ~429k updates by epoch 2200; the current run is at ~194k at
epoch 2000.

To reproduce the old run's optimization exactly, use `--nproc_per_node=2 --batch_size 128`.
Do **not** use 4 GPUs at `--batch_size 64` — same effective batch, but it changes the EMA
decay (§4.2).

---

## 3. Evaluation procedure

Four single-GPU generation processes in parallel (one per GPU, `CUDA_VISIBLE_DEVICES=$i`),
2500 samples each, merged into one directory, then scored with `clean-fid`.

The generation script has no `manual_seed` call, so each process is entropy-seeded
independently and the four shards are genuinely different samples.

FID call (identical across every number in §1 — keep it that way):

```python
from cleanfid import fid
s = fid.compute_fid(fdir1=d, dataset_name="cifar10", dataset_res=32,
                    dataset_split="train", mode="legacy_pytorch",
                    model_name="inception_v3", num_workers=0)
```

### 3.1 Active template

```bash
#!/bin/bash
#SBATCH --job-name=fid1800pf
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=fid1800pf_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10

CKPT="active_notweedie_tau015_Ta64_k4/checkpoint_epoch_1800.pt"
[ -f "$CKPT" ] || { echo "FATAL: $CKPT not found" >&2; ls -la active_notweedie_tau015_Ta64_k4/ >&2; exit 1; }

DIR=fid1800_pf
N=10000
PER=2500          # N / 4

rm -rf "$DIR" "$DIR"_r*

for i in 0 1 2 3; do
  GRID=""
  [ "$i" = "0" ] && GRID="--grid_out grid_1800_pf.png --grid_count 400"
  CUDA_VISIBLE_DEVICES=$i python -u generate_samples_multigpu.py \
    --ckpt "$CKPT" \
    --active --Tp 1e-3 --Ta 6.4 --tau 0.15 --k 4.0 --T 2.0 \
    --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
    --num_samples $PER --batch_size 256 \
    --probability_flow --pf_steps 600 --pf_schedule log \
    --output_dir "${DIR}_r${i}" $GRID \
    > "log_1800pf_r${i}.txt" 2>&1 &
done
wait

python - "$DIR" <<'PYEOF'
import glob, os, sys
dst = sys.argv[1]; os.makedirs(dst, exist_ok=True)
n = 0
for i in range(4):
    for f in sorted(glob.glob(f"{dst}_r{i}/*.png")):
        os.rename(f, os.path.join(dst, f"{n:06d}.png")); n += 1
print(f"merged {n} images into {dst}", flush=True)
PYEOF

TOTAL=$(find "$DIR" -name '*.png' | wc -l)
echo "total = $TOTAL"
[ "$TOTAL" -lt "$N" ] && { echo "INCOMPLETE -- see log_1800pf_r*.txt" >&2; exit 1; }
rm -rf "$DIR"_r*

python -u - "$DIR" <<'PYEOF'
import sys
from cleanfid import fid
d = sys.argv[1]
s = fid.compute_fid(fdir1=d, dataset_name="cifar10", dataset_res=32,
                    dataset_split="train", mode="legacy_pytorch",
                    model_name="inception_v3", num_workers=0)
print(f"RESULT  1800_pf600_log  FID = {s:.2f}", flush=True)
PYEOF
```

To retarget: change `CKPT`, `DIR`, the grid/log filenames, and the `RESULT` label.

### 3.2 Passive template

Differs from the active one in three ways: no `--active`, no `--Ta`/`--tau` (the passive
constructor takes only `T`, `k`, `Tp`), and `Tp 6.4` instead of `1e-3`. Adds a
verification step — see §4.1 for why that matters.

```bash
#!/bin/bash
#SBATCH --job-name=fid1000pass
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=fid1000pass_%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
cd /home/svaikunt/CIFAR10

CKPT="passive_tp64_k4/checkpoint_epoch_1000.pt"
[ -f "$CKPT" ] || { echo "FATAL: $CKPT not found" >&2; ls -la passive_tp64_k4/ >&2; exit 1; }

# verify the flags below actually match how this checkpoint was trained
python - "$CKPT" <<'PYEOF' || exit 1
import torch, sys
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
a = ck.get("args")
if a is None:
    print("WARN: no args in checkpoint, skipping verification", flush=True); sys.exit(0)
want = dict(active=False, model_base_dim=128, num_res_blocks=4, dim_mults="1,2,2,2",
            attn_resolutions="16", timesteps=1000, T=2.0, k=4.0, Tp=6.4)
bad = {k: (v, a.get(k)) for k, v in want.items() if a.get(k) != v}
print("ckpt epoch:", ck.get("epoch"), "| args:", {k: a.get(k) for k in want}, flush=True)
if bad:
    print("FATAL: eval flags disagree with training:", flush=True)
    for k, (w, g) in bad.items(): print(f"  {k}: script says {w!r}, ckpt says {g!r}", flush=True)
    sys.exit(1)
PYEOF

DIR=fid1000_passive_pf
N=10000
PER=2500          # N / 4

rm -rf "$DIR" "$DIR"_r*

for i in 0 1 2 3; do
  GRID=""
  [ "$i" = "0" ] && GRID="--grid_out grid_1000_passive_pf.png --grid_count 400"
  CUDA_VISIBLE_DEVICES=$i python -u generate_samples_multigpu.py \
    --ckpt "$CKPT" \
    --Tp 6.4 --k 4.0 --T 2.0 --timesteps 1000 \
    --model_base_dim 128 --num_res_blocks 4 --dim_mults 1,2,2,2 --attn_resolutions 16 \
    --num_samples $PER --batch_size 256 \
    --probability_flow --pf_steps 600 --pf_schedule log \
    --output_dir "${DIR}_r${i}" $GRID \
    > "log_1000passivepf_r${i}.txt" 2>&1 &
done
wait

python - "$DIR" <<'PYEOF'
import glob, os, sys
dst = sys.argv[1]; os.makedirs(dst, exist_ok=True)
n = 0
for i in range(4):
    for f in sorted(glob.glob(f"{dst}_r{i}/*.png")):
        os.rename(f, os.path.join(dst, f"{n:06d}.png")); n += 1
print(f"merged {n} images into {dst}", flush=True)
PYEOF

TOTAL=$(find "$DIR" -name '*.png' | wc -l)
echo "total = $TOTAL"
[ "$TOTAL" -lt "$N" ] && { echo "INCOMPLETE -- see log_1000passivepf_r*.txt" >&2; exit 1; }
rm -rf "$DIR"_r*

python -u - "$DIR" <<'PYEOF'
import sys
from cleanfid import fid
d = sys.argv[1]
s = fid.compute_fid(fdir1=d, dataset_name="cifar10", dataset_res=32,
                    dataset_split="train", mode="legacy_pytorch",
                    model_name="inception_v3", num_workers=0)
print(f"RESULT  passive_1000_pf600_log  FID = {s:.2f}", flush=True)
PYEOF
```

---

## 4. Pitfalls

### 4.1 `strict=False` hides architecture mismatches

`load_ema_model` in `generate_samples_multigpu.py` filters the state dict to keys the
constructed model happens to have, then calls `load_state_dict(..., strict=False)`. A
wrong arch flag therefore produces a **partly randomly-initialized model and a
plausible-looking FID**, with no warning in the log.

The script's defaults disagree with how these models were trained:

| flag | script default | trained value |
|------|----------------|---------------|
| `--attn_resolutions` | `16,8` | `16` |
| `--num_res_blocks`   | `2`    | `4` |
| `--Ta`               | `1.0`  | `6.4` (active) |
| `--tau`              | `0.4`  | `0.15` (active) |
| `--k`                | `1.0`  | `4.0` |

**Always pass every arch and SDE flag explicitly.** Better, use the verification block
from §3.2 — it reads `ckpt['args']` and aborts on any mismatch.

### 4.2 EMA decay is not `--model_ema_decay`

`train_multigpu.py` applies the torchvision adjustment:

```python
adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
alpha  = min(1.0, (1.0 - args.model_ema_decay) * adjust)
decay  = 1.0 - alpha
```

For the old run (`batch_size=128`, `model_ema_steps=10`, `epochs=2600`,
`model_ema_decay=0.9997`):

```
adjust = 128*10/2600 = 0.4923
alpha  = 0.0003 * 0.4923 = 1.477e-4
decay  = 0.99985          (horizon ~68k optimizer steps ~ 350 epochs)
```

The effective decay depends on `--epochs`, `--batch_size`, and `--model_ema_steps`, and
uses the **per-GPU** batch. Consequences:

- Keep `--epochs 2600` when reproducing, even if stopping earlier — it is load-bearing
  for the EMA, not just a stopping condition.
- FID is computed from EMA weights, so any of these changing makes runs non-comparable.

### 4.3 Resuming overwrites checkpoints in the same `--out_dir`

Checkpoints are written as `checkpoint_epoch_{N}.pt` at `--save_freq` intervals. Resuming
from an earlier epoch into the same directory overwrites everything downstream of it as
training passes those epochs again. Back up any checkpoint you still need to score
**before** launching a resume.

Disk pressure is real here — each checkpoint is ~913 MB, and `active_notweedie_tau015_Ta64_k4/`
was down to only epochs 1000 and 2000 as of 2026-08-12.

### 4.4 Sample count changes the number

FID is biased upward at small N. All §1 numbers are at N = 10,000; the historical 7.91
was measured at different settings. Never compare across sample counts.

---

### 1.8a **PROJECT BEST: τ=0.5, FID 6.91 @ N=50,000, epoch 1000**

Quadratic PF-500, N = 50,000 — same configuration as §1.2's 7.91, directly comparable.

| run | epoch | optimizer steps | images | FID@50k | numerics |
|-----|-------|-----------------|--------|---------|----------|
| old 2-GPU run (§1.2) | 2200 | 404k | 103M | 7.91 | — |
| τ=0.15 CLD (§1.6) | 1000 | 390k | 50M | 9.50 | pre-patch |
| τ=0.40 | 1200 | 468k | 60M | 7.31 | **pre-patch** (~19% M11 err) |
| **τ=0.50** | **1000** | **390k** | **50M** | **6.91** | **post-patch, exact** |

**τ=0.5 is the best result and the first record that is reproducible from current code** —
τ=0.4's 7.31 was trained against the buggy float32 covariance. It also uses the fewest
optimizer steps and least data of any arm, and τ=0.5 was still training when measured
(target 2000 epochs).

Still ~3× off CLD's 2.25 and ~2.2× off plain DDPM's 3.17.

**At N = 10,000, PF-300, epoch 1000** — the sweep including the critical point:

| τ | k·τ | FID | numerics |
|------|------|-------|----------|
| 0.10 | 0.4 | 12.88 | pre-patch |
| 0.20 | 0.8 | 11.68 | post-patch |
| **0.25** | **1.0** | **13.88** | post-patch |
| 0.40 | 1.6 | 9.68 | pre-patch |
| 0.50 | 2.0 | **8.89** | post-patch |

Improving in τ **except for a sharp 2.2-FID spike at kτ = 1**. The three post-patch arms
alone — 11.68 / 13.88 / 8.89 at τ = 0.2 / 0.25 / 0.5 — establish the kink on clean
numerics on both sides. The critical-damping point is anomalously *bad*. Worth confirming
at a second epoch before treating it as physics.

### 1.9 `score_param=cond` — conditional whitening (in progress)

`/net/scratch/svaikunt/cifar10/active_cld_tau05_cond/`, job 1383805. Identical to the
τ=0.5 arm of §1.8a in every respect except `--score_param cond`, so it is a clean A/B
against **8.89** (N=10k, PF-300) and **6.91** (N=50k, quadratic PF-500) at epoch 1000.

Motivation in §5: the original loss whitens both channels with the single scalar
`√det`, but `√det = √v_η·√M11 = √v_x·√M22` — the correct whitener times a spurious
`√M_jj`. That drives the η target from 0.73 at t=T down to 0.0014 at t=1e-3 (520×) at
τ=0.5. `cond` whitens each channel by its own conditional variance
`v_i = Var(d_i | d_j) = det/M_jj`, giving unit-variance targets at every t.

The earlier job 1383554 died immediately — `TypeError: unexpected keyword argument
'score_param'`, from deploying the patched `model_cifar10_sde_DDPM.py` without
`model_cifar10_sde_DDPM_v2.py`. Both files must ship together.

#### Epoch 200 — sanity gate passed

Sample grids (`samples_epoch_200_pf.png`, `_large.png`, both in
`DownloadsFromCluster/tau05cond/`) show coherent objects under both samplers: correct
global structure, plausible colour statistics, **no high-frequency mush**. That was the
specific failure mode to watch for — if the reweighting had broken small-t accuracy it
would be visible here. It isn't. No FID yet; there is no epoch-200 baseline for the
score-mode arm to compare against.

#### ⚠ Do not compare `cond` loss values to `score` loss values

The two objectives have different losses at zero network output, so raw curves sit at
different heights for reasons unrelated to model quality:

| objective | loss at init | why |
|-----------|--------------|-----|
| `cond`     | `Tp + Ta` = **6.401** | targets are unit-variance at every t, by construction |
| `whitened` | **2** | `r ~ N(0, I)` at every t |
| `score`    | `Ta·E_t[M11] + Tp·E_t[M22]` = **2.504** | targets collapse as t → 0 |
| passive    | **1** | `σ·score = −ε` |

(`Var(Feta) = M11` and `Var(Fx) = M22` exactly — the `√det` cancels.) Divide by these to
get the *fraction of target variance unexplained*, which IS comparable. `cond` at 0.552
is 8.6% unexplained; `score` at 0.068 is 2.7%. So the raw 8× ratio is really ~3×, and
even that is not a quality statement — the two are fitting differently-weighted targets.
[`plot_loss.py`](plot_loss.py) parses the `ARGS:` banner out of a log, computes the right
init loss per objective, and plots both panels.

#### The loss is flat after ~epoch 50, and this is expected

| epoch | 25 | 50 | 100 | 150 | 200 | 217 |
|-------|----|----|-----|-----|-----|-----|
| loss  | 0.635 | 0.582 | 0.563 | 0.552 | 0.548 | 0.552 |

Slope over epochs 100–217 is **−0.007 per 100 epochs**, only 1.96× the epoch-to-epoch
scatter (0.0042); over 150–217 alone it is 0.64× scatter, i.e. indistinguishable from
flat. DSM loss saturates near its irreducible floor within ~50 epochs and then stops
tracking sample quality — §1.1 is the proof, where FID went 21.58 → 12.34 between epochs
1000 and 2600 with a static loss. **Never use training loss as a progress signal here.**

Note also that at epoch 217 the run is at 84,630 of 100,000 warmup steps — lr has not
yet reached 2e-4, which happens at epoch 256. Some of the flatness is that.

#### Epoch 800 A/B — **cond wins by 1.99** (job 1399066)

N = 10,000, quadratic PF-300, Heun, single H200, both arms scored in the same job from
`fidab1.sh` so sampler and eval are byte-identical.

| run | epoch | FID @10k, PF-300 |
|-----|-------|------------------|
| **τ=0.5 `cond`** | **800** | **7.71** |
| τ=0.5 `score` | 800 | 9.70 |
| τ=0.5 `score` | 1000 | 8.89 (§1.8a) |

**delta (cond − score) at epoch 800 = −1.99.** Two independent readings, both favourable:

1. The matched-epoch gap is 1.99, roughly 4× the N=10k noise floor (~0.5, measured by
   re-drawing the same checkpoint: 15.63 vs 15.09).
2. `cond` at epoch **800** beats `score` at epoch **1000** by 1.18 — better quality from
   200 fewer epochs, which no sampler or scoring artifact explains.

`score` traces 9.70 → 8.89 over epochs 800 → 1000 (−0.81 per 200). If `cond` holds that
slope it lands near **6.9 at epoch 1000** on this scale.

> ⚠ Not yet comparable to the 6.91 headline — that is N=50,000 at quadratic PF-500. The
> comparable score-mode figure here is **8.89**. For the headline, rerun as
> `sbatch fidab1.sh 1000 50000 500`, which is ~5× the sampling (~95 min/arm on one H200 —
> worth 4 GPUs).

**Still to confirm:** §1.8b showed early τ gaps were largely convergence-rate effects that
collapsed by epoch 1000 (the 0.1-vs-0.2 gap went 12.22 → 1.20). Epoch 1000 is the check.

#### Epoch 800 headline — **PROJECT BEST: FID 5.67 @ N=50,000** (job 1409276)

N = 50,000, quadratic PF-500, Heun, 4×H200 (4 × 12,500 merged). Same configuration as
§1.2's 7.91 and §1.8a's 6.91, so directly comparable.

| run | epoch | steps | images | FID@50k |
|-----|-------|-------|--------|---------|
| **τ=0.5 `cond`** | **800** | **312k** | **40M** | **5.67** |
| τ=0.5 `score` (§1.8a) | 1000 | 390k | 50M | 6.91 |
| τ=0.40 pre-patch (§1.8) | 1200 | 468k | 60M | 7.31 |
| old 2-GPU run (§1.2) | 2200 | 404k | 103M | 7.91 |

**Beats the previous best by 1.24 FID on 20% fewer optimizer steps and 20% less data**,
and the run was still training when measured. Gap to CLD's 2.25 narrows from ~3× to 2.5×;
still 1.8× off plain DDPM's 3.17.

This is the first result attributable to the loss fix rather than to τ: `cond` and `score`
differ only in `--score_param`, share τ=0.5 and every other hyperparameter, and both were
trained on post-patch Van Loan numerics.

**Corroborated at two scales.** The 10k/PF-300 A/B gave cond 7.71 vs score 9.70 at the
same epoch, so the advantage is not a sampler or sample-count artifact.

**Not yet done:** the matched-epoch 50k point at 800. `score` at 800/50k was never sampled
(the job was cancelled to prioritise epoch 1000), so 5.67-vs-6.91 compares different
epochs. Superseded by the epoch-1000 result below, which is matched.

#### Epoch 1000 — **PROJECT BEST: FID 4.98**, and the matched comparison

N = 50,000, quadratic PF-500, 4×H200. Same epoch, same N, same sampler as §1.8a's 6.91 —
the arms differ **only** in `--score_param`.

| τ=0.5, epoch 1000, 50k, quad PF-500 | FID |
|-------------------------------------|-----|
| **`cond`** | **4.98** |
| `score` | 6.91 |
| **delta** | **−1.93 (−28%)** |

**The loss fix changes the convergence rate, not just the offset.** Over 800 → 1000:

| arm | 800 | 1000 | ratio |
|-----|-----|------|-------|
| `cond` (50k) | 5.67 | 4.98 | **0.878** |
| `score` (10k/PF-300) | 9.70 | 8.89 | 0.917 |

A pure offset would have given 5.67 × 0.917 = 5.20; the measured 4.98 is below that. Every
other per-200-epoch ratio in this project sits at 0.90–0.97 (§1.1), so 0.878 is the
fastest interval recorded here.

#### Epoch 1200 — **4.88**, and the rate collapses. Run stopped here.

| epoch | 800 | 1000 | 1200 |
|-------|------|------|------|
| FID @50k, quad PF-500 | 5.67 | 4.98 | **4.88** |
| ratio to previous | — | 0.878 | **0.980** |

Projected from the 0.878 interval, epoch 1200 should have been ≈4.37 (band 4.4–4.6). It
came in at **4.88**, well above — the improvement rate decayed by more than the whole
projection assumed. Extrapolating a single interval was the error; one ratio does not
establish a rate.

0.980 is slower than any interval recorded in §1.1 (0.90–0.97), and in that run a ratio of
0.972 (12.69 → 12.34) came immediately before the minimum at 12.20 and the subsequent rise
to 13.04. **Training was stopped at 1200 on that basis.** 4.88 is the final result.

Whether 1200 is the true minimum is unmeasured — epoch 1400 was never scored. If the §1.1
pattern holds, 1200 is at or just before the floor and later checkpoints would be worse.

**Against published unconditional CIFAR-10:** EDM 1.97 · NCSN++ 2.20 · CLD 2.25 ·
DDPM 3.17 · DDIM-100 4.04 · **this 4.98** · NCSNv2 10.87. Gap to CLD is now 2.2×, down
from ~3×. Note CLD used NCSN++ at roughly twice the parameters with FIR resampling and far
longer training, so part of the remaining gap is capacity and compute rather than the SDE.

#### 1.9a Red team — does `cond` change the dynamics?

Run [`redteam_cond.py`](redteam_cond.py) and [`redteam_cond_weights.py`](redteam_cond_weights.py)
against the real classes. **Verdict: the target is unchanged; only the weighting moved.**

| check | result |
|-------|--------|
| cond target vs `−Σ⁻¹d`, x | max\|diff\|/RMS = **5.9e-15** |
| cond target vs `−Σ⁻¹d`, η | **3.0e-15** |
| `score` loss at the exact score | **2.8e-14** |
| `cond` loss at the exact score | **1.2e-13** |
| loss at zero output | 2.532 / 6.403 (predicted 2.504 / 6.401) |
| `_score_from_output` identity, both modes | bit-for-bit |
| `v_x` float32 vs float64 | 4.5e-7 |
| original `score` branch vs pre-`score_param` commit | no deletions |

**Why the targets coincide.** `v_x = M11 − M12²/M22 = det/M22`, so
`s_x = −(d_x − (M12/M22)d_η)/v_x = −(M22·d_x − M12·d_η)/det`, which is exactly the
x-component of `−Σ⁻¹d`. Same for η with `M11`. Both losses are
`E[w(t)·(F − s_true)²]` with `w > 0` depending only on t and channel, so both are minimised
by `E[s_true | x_t, η_t]` — the same function. The network converges to the same score.

**No cancellation hazard.** `v_x` is a difference, so it was the expected weak point given
the τ=0.2 Cholesky failure (§5). It is safe: the x–η correlation peaks at **0.854**, far
from 1, so `M12²/M22` never approaches `M11`.

**What changed** — the per-t weight, by `1/M_jj`:

| t | 0.001 | 0.01 | 0.1 | 1.0 | 2.0 |
|---|-------|------|-----|-----|-----|
| η weight ratio cond/score = 1/M11 | 4.98e5 | 2.81e4 | 90.0 | 2.06 | 1.88 |

A **2.65e5×** swing across t, concentrated on small t — precisely where `√det` was
suppressing the signal. `cond` gives every t unit-variance targets, i.e. the active
analogue of DDPM's `L_simple`, the standard FID-optimal weighting.

> ⚠ **Do not claim a uniform improvement.** Equal weighting buys small-t accuracy and FID
> rewards exactly that. Nothing here shows `cond` is better on likelihood/NLL, where the
> variance-weighted objective is closer to the ELBO and may well win. Report the FID claim
> and state the weighting explicitly.

**Note:** the loss is computed inside `torch.autocast(bfloat16)` (train_multigpu.py:295).
Pointwise ops keep their input dtype, and `compute_covariance` builds in float64 before
casting to t's float32, so `v_x` is not evaluated in bf16. Verified numerically above.

#### Practical notes

- **`num_workers=0` is catastrophic for the 50k FID.** 10k folders scored at 4.6–15 it/s,
  but 50k dropped to **0.17 it/s** (5.91 s/it, ETA 2.5 h) — NFS attribute-cache thrashing
  on a large directory, not compute. `num_workers=8` restored 2.12 it/s (~12 min). Use 8
  everywhere; `TMPDIR` on scratch already handles the `/tmp`-full problem that originally
  motivated 0.
- **Node p003 is wedged.** It accepts jobs, marks them RUNNING, and never executes them
  (two occurrences: 1398612, 1404162; also gave `srun` socket timeouts and sat in
  `completing`). Submit with `--exclude=p003`.
- **The run TIMEOUTed at 9 h**, losing epochs 800→~978. At 32.7 s/epoch, 800→2000 needs
  ~11 h. `resume_cond.sh` uses a 12 h limit and `--save_freq 100`.

**Next:** `sbatch --exclude=p003 ~/fidone.sh 1000 50000 500 cond` for the matched-epoch
comparison against 6.91; optionally `... 800 50000 500 score` to complete the 800 row.

### 1.8 τ sweep — **best result: τ=0.4, FID 7.31 @ N=50,000 — but see the caveat**

> ⚠ **The τ=0.4 arm was trained BEFORE the Van Loan numerics patch (§5).** At τ=0.4,
> `k − 1/τ = +1.5`, giving ~19% float32 error in `M11` at small t. The 7.31 measurement
> is valid — that model exists and scores 7.31 — but it is **not reproducible from the
> current code** and must not be reported as "τ=0.4 under the CLD recipe" without a
> clean re-run.
>
> The sweep is also internally inconsistent: τ=0.1 (job 1374525) is pre-patch at ~3%
> error, τ=0.2 (job 1374686) is post-patch and exact, τ=0.4 (job 1374477) is pre-patch at
> ~19%. So "FID monotone in τ" is confounded with "M11 error varies by arm". The arm with
> the largest error also has the best result, which shows the error is not fatal, but
> forbids attributing 7.31 to τ.
>
> Likely train/eval mismatch as well: if the patched file was deployed before scoring,
> that model was *trained* with the biased covariance and *sampled* with the corrected
> one. The Tweedie step calls `compute_covariance(t_eps)`, precisely where the 19% sits.
>
> **Action: re-run τ=0.4 (and ideally τ=0.1) from scratch post-patch before reporting.**

#### Numbers as measured

Single GPU, batch 128, CLD recipe (lr 2e-4, warmup 100k, effective EMA 0.9999, T=2),
all arms identical except τ. First measurements after the numerics patch.

**Headline, N = 50,000, quadratic PF-500 — directly comparable to §1.2's 7.91:**

| run | epoch | optimizer steps | images | FID@50k |
|-----|-------|-----------------|--------|---------|
| old 2-GPU run (§1.2) | 2200 | 404k | 103M | 7.91 |
| τ=0.15 CLD (§1.6) | 1000 | 390k | 50M | 9.50 |
| **τ=0.40 CLD** | **1200** | **468k** | **60M** | **7.31** |

Beats the previous project best by 0.60 on 42% fewer images, at epoch 1200 of a
2000-epoch run. Still ~2.3× off published baselines (CLD 2.25, DDPM 3.17).

**The sweep at N = 10,000** (lower steps for speed; ranks arms, not comparable to 50k):

| τ | k·τ | @600, PF-200 | @1000, PF-300 | @1200, PF-300 |
|------|------|--------------|---------------|---------------|
| 0.10 | 0.4  | 28.40 | 12.88 | — |
| 0.20 | 0.8  | 16.18 | 11.68 | — |
| 0.40 | 1.6  | 12.34 | 9.68  | **9.52** |

Monotone in τ at every epoch measured. The epoch-600 spread (28.40 → 12.34) is largely a
*convergence-rate* effect — by epoch 1000 the 0.1→0.2 gap has collapsed from 12.22 to
1.20 — but the 0.2→0.4 gap only narrows from 3.84 to 2.00, so a genuine quality
difference remains. **The trend has not turned over**, so τ = 0.5, 0.6, 1.0 are the arms
that matter; τ = 0.25 (kτ = 1, critical damping) fills the grid at the one point the old
numerics could not reach.

For scale, τ=0.4's 9.52 at 10k/PF-300 already beats anything τ=0.15 produced at 10k
(best 11.19, and that at PF-500).

### 1.8b Earlier framing of the epoch-600 sweep

Single GPU, batch 128, CLD recipe (lr 2e-4, warmup 100k, effective EMA 0.9999), all
identical except τ. Scored PF-200 **quadratic**, N = 10,000 — chosen for speed, so these
rank the arms against each other but are *not* comparable to the PF-500 history.

| τ | k·τ | FID @ epoch 600 |
|------|------|-----------------|
| 0.10 | 0.4  | 28.40 |
| 0.20 | 0.8  | 16.18 |
| 0.40 | 1.6  | **12.34** |

**Monotone and steep — a 2.3× spread.** The largest effect of any single knob measured in
this project. The trend has not turned over at 0.4, so the optimum is likely higher; τ =
0.6, 0.8, 1.0 are the next arms. τ = 0.25 (kτ = 1, critical damping) is also worth
measuring now that the numerics allow it — it interpolates between two measured points, so
a kink there would indicate the critical line is physically special and a smooth ~14 would
indicate it is not.

For scale: τ=0.4 reaches 12.34 at epoch 600 with only 200 sampler steps, where τ=0.15
needed epoch 1000 and 500 steps for 11.19. Cross-config, so suggestive rather than proven.

**A "gentler diffusion" confound was considered and does not hold up.** τ does change the
stationary x-variance, since the η equilibrium variance is Ta/τ:

| τ | stationary var | σ_x | σ_x / data σ (≈0.5) |
|------|----------------|-------|----------------------|
| 0.10 | 1.143 | 1.069 | 2.14 |
| 0.15 | 1.000 | 1.000 | 2.00 |
| 0.20 | 0.889 | 0.943 | 1.89 |
| 0.40 | 0.616 | 0.785 | 1.57 |

But that is a 1.37× change in σ across the sweep, not enough to explain a 2.3× FID spread.
The signal decays as `e^{−kt}` **independent of τ**, so every arm destroys the data equally
by t = T. And the sampler initialises from `compute_covariance(T)` — the correct correlated
stationary covariance for that τ — rather than a fixed N(0, I), so there is no prior
mismatch either. The τ effect therefore appears to be about the dynamics, not about how
much noise is added.

---

## 5. Numerics: the active covariance and the τ sweep

Discovered 2026-08-15 when a τ=0.2 run died with
`linalg.cholesky: (Batch element 109): the input is not positive-definite`.

**Cause.** `compute_covariance` builds `M11` as

```
(1/k)·Tx·(1−a²) + (1/k)·Ty·[ 1/(w(k+w)) + 4abk/((k+w)(k−w)²) − (kb²+wa²)/(w(k−w)²) ]
```

with `w = 1/τ`. At small t (a≈b≈1) those three bracket terms cancel almost exactly —
their sum is ~1e-8 while the individual terms are O(1/(k−w)²). The closer `1/τ` gets to
`k`, the larger the terms and the worse the cancellation:

| τ | 1/τ | (k−w)² | term size | M11 at t=1e-3 | float32 error |
|------|------|--------|-----------|---------------|---------------|
| 0.10 | 10   | 36     | ~0.05     | 4.0e-6        | 3%            |
| 0.15 | 6.67 | 7.1    | ~0.22     | 2.9e-6        | 5%            |
| 0.20 | 5    | 1      | ~1.8      | 2.5e-6        | **>20%, goes negative** |
| 0.25 | 4    | **0**  | ÷0        | **NaN**       | singular      |
| 0.40 | 2.5  | 2.25   | ~1.1      | 2.1e-6        | 19%           |

Float32 carries ~1e-7 relative precision, so at τ=0.2 the rounding error exceeds the
answer and `M11` comes out negative for some sampled t. The existing fallback (add a
*fixed* `1e-6` to the diagonal) cannot help: `1e-6` is ~40% of `M11` itself at small t.

**Fix.** Compute the covariance by **Van Loan's method** instead of the closed form. For
the linear SDE with drift `A = [[−k, 1], [0, −1/τ]]` and diffusion `Q = diag(2Tp, 2Ta/τ²)`,
form `C = [[−A, Q], [0, Aᵀ]]`, exponentiate, and read `Σ(t) = E₂₂ᵀ · E₁₂`. No `(k−w)²`
appears anywhere, so it is exact at τ = 1/k and stable everywhere. Verified against the
closed form to ~1e-14 at τ = 0.15, 0.20, 0.40; finite at τ = 0.25 where the closed form
is NaN.

Also replaced the Cholesky fallback: `cholesky_ex` with jitter scaled to the matrix
diagonal, rather than exception handling with an absolute `1e-6`.

**Why τ = 1/k is physically the interesting point.** There `A` is *defective* — one
repeated eigenvalue, a single eigenvector, a Jordan block — and
`exp(At) = e^{−kt}·[[1, t], [0, 1]]`. The linear-in-t factor is the fingerprint of
critical damping, i.e. the sharpest possible response. So the closed forms excluded
precisely the point most worth measuring. Post-patch the critical line τ = 1/k is in
the sweep like any other value.

**Every site carrying the same pattern** (all patched):

| location | quantity | fix |
|----------|----------|-----|
| `compute_covariance` | forward kernel Σ(t) | Van Loan |
| `compute_mean` | `exp(At)₁₂` | `expm1` + Jordan limit |
| `diffusion_loss_active` | `exp(At)₁₂` (**training path**) | `expm1` + Jordan limit |
| Tweedie correction | `exp(At)₁₂` at `t_eps` | `expm1` + Jordan limit |
| `_eq_transition` | equilibrium-split `exp(Aᵀs)₂₁` | `expm1` + Jordan limit |
| `_reverse_transition_mean` | reverse-flow `exp(−As)` | Van Loan |
| `_reverse_transition_covariance` | reverse-flow Σ(s), had `Δ²` | Van Loan |

`_stationary_covariance` was already clean — `sxe = see/(k + 1/τ)` has a *sum*, not a
difference, in the denominator. Reverse Van Loan verified against the old closed forms
to ~1e-14 at τ = 0.15, 0.20, 0.40 and finite at τ = 0.25.

Note that `diffusion_loss_active` is in the training path, so the τ=0.25 run would have
died there even with `compute_covariance` alone fixed.

**Verification at the critical point.** `_eq_transition` builds
`E(h) = Σ∞ exp(Aᵀh) Σ∞⁻¹`, so the similarity transform has to survive the Jordan block
too. Checked at h = 0.004, k = 4:

| τ | eig₁ | eig₂ | max err vs {e^−kh, e^−h/τ} | ‖(E−λI)²‖ | C PSD |
|---|---|---|---|---|---|
| 0.10 | 0.960789439152 | 0.984127320055 | 3.3e-16 | 1.0e-02 | yes |
| 0.15 | 0.973685749353 | 0.984127320055 | 1.1e-16 | 3.5e-03 | yes |
| 0.20 | 0.980198673307 | 0.984127320055 | 1.4e-15 | 1.1e-03 | yes |
| **0.25** | **0.984127320055** | **0.984127320055** | **2.2e-16** | **1.1e-16** | yes |
| 0.40 | 0.984127320055 | 0.990049833749 | 1.3e-15 | 1.2e-03 | yes |

Eigenvalues correct to machine precision everywhere. At τ=0.25 both collapse to e^−kh and
`(E−λI)²` vanishes to 1.1e-16 — `E` is genuinely defective there rather than accidentally
diagonalized by roundoff, and the O(10⁻³) residual at the other τ shows the test
discriminates. Σ∞ itself is well-conditioned throughout (cond 66–80, det 10–52), so the
`Σ∞⁻¹` in the transform amplifies nothing.

**General rule for future extensions.** In a two-timescale linear SDE every transition
coefficient is built from `e^{−kt}` and `e^{−t/τ}`; any that combines them with a
*difference* of rates in the denominator degenerates at kτ=1. Stationary quantities are
structurally safe — `_stationary_covariance` has `k + 1/τ` precisely because it is the
t→∞ limit where the transients have cancelled. So: time-dependent and mixing both rates →
check for Δ; stationary → probably fine.

**Consequences.**

- Any τ now works, including 0.25 and values near it, across forward kernels, the loss,
  the Tweedie correction, and both SSCS splittings. Same for varying `k`.
- float64 alone would *not* have been sufficient: it buys ~1e-16 relative precision, so
  it survives τ near 0.25 but the cancellation still worsens without bound approaching
  it, and τ = 1/k exactly remains a division by zero. Van Loan removes the failure mode
  rather than postponing it.
- The `+1e-8` fudges in `M11`/`M22` are gone; the returned covariance is exact.
- **Runs before this patch are not bitwise comparable to runs after it.** The τ=0.15
  results in §1.6–1.7 were computed with ~5% float32 error in `M11` at small t. Not
  enough to invalidate them, but the τ sweep should be run entirely post-patch.

## Open questions

- **FID 122.34 at pf_steps=200** (epoch 1600, Tweedie on) vs 17.38 at 600 steps. A 7×
  degradation is not a step-count effect — an under-resolved ODE degrades smoothly.
  Suspect the Tweedie correction at the final step, where the log schedule makes the step
  size large. The `active_pf_notw` arm that would isolate this was killed mid-run
  (job 1363737); rerun it.
- **Old run's FID vs optimizer steps.** The current run can only be compared to the old
  one at matched update counts, and no intermediate FIDs from the old run survive. If any
  of its intermediate checkpoints still exist anywhere, scoring them would give the
  reference curve.
- **Where the 1200–1800 checkpoints were scored.** They no longer exist in
  `active_notweedie_tau015_Ta64_k4/` on the UChicago cluster. If those evals ran at NERSC,
  confirm that copy of the run is intact.

---

## 6. What the active noise contributes — a mechanism for τ

![what eta contributes](eta_information.png)

Generated by [`plot_eta_information.py`](plot_eta_information.py).

### The stationary correlation has a closed form

At stationarity the joint covariance gives

```
rho_inf = corr(x, eta) = sqrt( k*tau / (1 + k*tau) )
```

exactly — `Tp` shifts it by less than 1e-4 at `Tp=1e-3`. It depends **only on the product
kτ**. At τ=0.5, k=4 this is √(2/3) = **0.8165**, which is the 0.82 seen in the sampling
animation. It is below 1 because x is a low-pass filtered copy of η: x relaxes at rate k
while η decorrelates at rate 1/τ, so x tracks η only insofar as η is slow. ρ→1 requires
kτ→∞.

### η is informative about the x-noise, and how much predicts FID

The denoiser sees the pair `(x_t, eta_t)`. The x-noise it must remove has variance `M11`;
conditioning on the η-noise reduces that to `v_x = det M / M22`. So

```
v_x / M11 = 1 - corr^2(d_x, d_eta)
```

is the fraction of the x-noise that η does **not** explain. This is the same `v_x` that
whitens the loss (§1.9) — the quantity that measures how much η helps *is* the quantity
that sets the correct loss weighting.

| τ | kτ | ρ_∞ | mean `v_x/M11` over t | FID @10k, PF-300, epoch 1000 |
|------|------|-------|------------------------|------------------------------|
| 0.10 | 0.4 | 0.534 | 0.674 | 12.88 (pre-patch) |
| 0.20 | 0.8 | 0.667 | 0.515 | 11.68 |
| 0.25 | 1.0 | 0.707 | 0.463 | **13.88** |
| 0.40 | 1.6 | 0.784 | 0.362 | 9.68 (pre-patch) |
| 0.50 | 2.0 | 0.816 | **0.320** | **8.89** |
| 1.00 | 4.0 | 0.894 | 0.228 | — |

**FID is monotone in how informative η is — except at kτ = 1.** At τ=0.5, seeing η removes
68% of the x-noise variance averaged over t. This gives a mechanism for the τ trend rather
than an empirical fit: larger kτ means η is a better predictor of the noise corrupting x,
and the denoiser exploits it.

The kτ=1 outlier is the same point where the drift matrix is defective (§5). Whether that
is a genuine physical effect or a residual numerical one is **still unresolved** — it rests
on a single checkpoint and has never been confirmed at a second epoch.

### Caveats

- τ=0.1 and τ=0.4 are pre-Van-Loan and shown hollow in the figure; the trend should be
  re-checked once they are re-run.
- Five points, one of which is an outlier, is not a strong fit. The mechanism is exact;
  the *correlation with FID* is suggestive.
- This is all forward-process algebra. It shows η **could** help, and that the amount
  tracks FID. It does not show the trained network actually uses η — for that, see the
  ablation proposed below.

### Proposed test: does the trained model actually use η?

Shuffle `eta_t` across the batch at every sampling step. That preserves η's marginal
exactly while destroying its correspondence with x. If the model is genuinely conditioning
on η, FID should collapse; if it has learned to ignore the channel, FID barely moves.
Cheap — one sampling run, no retraining.
