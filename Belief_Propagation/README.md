# Belief Propagation on Hierarchical Trees

Exact belief propagation (BP) on a tree-structured hierarchical generative model, comparing how well passive vs active diffusion at the leaves preserves information about the latent tree structure.

---

## What it does

### Generative model

A random tree with:
- **Alphabet size** `v` — number of discrete symbols at each node
- **Branching factor** `s` — each parent has `s` children
- **Depth** `L` — number of levels (root at level `L`, leaves at level 0)
- **Production rules** `m` — each parent symbol maps to `m` allowed child tuples (randomly generated, disjoint across parents)

States are sampled top-down: the root draws a symbol uniformly, then each node's children are drawn from its `m` production tuples.

### Diffusion at the leaves

Two SDE processes corrupt the leaf one-hot embeddings before BP runs:

- **Passive** — OU process: `dx = -x dt + √(2T) dW`  
  Exact posterior `p(x₀ | xₜ)` is a softmax over the diffused coordinates.

- **Active** — coloured-noise SDE:
  ```
  dx  = (-k x + η) dt + √(2T) dW_x
  dη  = (-η/τ)    dt + (1/τ)√(2Ta) dW_η
  ```
  Exact posterior `p(x₀ | xₜ, ηₜ)` derived analytically from the joint Gaussian transition kernel.

### Belief propagation

Sum-product BP runs in two passes:
1. **Upward pass** — leaf posteriors are propagated to the root using the production rules as factor nodes
2. **Downward pass** — a uniform root prior is refined back down to the leaves

Node-level marginals are computed as the product of upward and downward messages. Accuracy at each tree level is the fraction of nodes whose MAP estimate matches the true sampled state.

### Comparison

Three conditions are evaluated over a log-spaced grid of diffusion times `t`:

| Condition | Description |
|---|---|
| **Passive** | Standard OU noise at temperature `T` |
| **Active** | Active noise with `Tp, Ta, k, τ` |
| **Equivalent passive** | Passive at temperature `T_equiv = Ta / (k (1 + τ k))`, matching the active stationary variance |

This isolates whether the coloured-noise structure of the active process provides an information advantage beyond a simple temperature rescaling.

---

## Files

| File | Purpose |
|---|---|
| `Belief_prop.py` | Full BP implementation: tree generation, diffusion, posteriors, BP passes, accuracy, plotting |
| `MakePlots.ipynb` | Load saved results and produce publication-quality plots |

---

## Running

### Full simulation

Runs all three conditions (100 trials each, 40 log-spaced time points) and saves results:

```bash
python Belief_prop.py
```

Output: `bp_accuracy_results.npz` with arrays `acc_diffusion`, `acc_active`, `acc_diffusion_equiv`, and the simulation parameters.

Default hyperparameters (edit `main()` to change):

| Parameter | Value | Meaning |
|---|---|---|
| `v` | 32 | Alphabet size |
| `s` | 2 | Branching factor |
| `L` | 10 | Tree depth |
| `m` | 8 | Production rules per parent |
| `T` | 1.0 | Passive temperature |
| `Tp` | 0.001 | Active passive temperature |
| `Ta` | 1.0 | Active noise strength |
| `tau` | 2.0 | Active persistence time |
| `k` | 1.0 | Spring constant |
| `ntrials` | 100 | Trials averaged per time point |

### Plotting

Open `MakePlots.ipynb` after the simulation has finished. It produces two figures:

- **`Root_bp_accuracy_results.pdf`** — root-level accuracy vs time for passive, active, and equivalent-passive
- **`Bp_accuracy_results_active.pdf`** — accuracy at every tree level vs time for the active condition (coloured by level)

```bash
jupyter notebook MakePlots.ipynb
```

---

## Key functions (`Belief_prop.py`)

| Function | Description |
|---|---|
| `generate_rules(v, s, m, L)` | Randomly generate disjoint production rules for a tree |
| `sample_states_top_down(...)` | Sample a full tree realization top-down |
| `diffuse_forward(...)` | Apply passive OU diffusion to a leaf one-hot embedding |
| `diffuse_forward_active(...)` | Apply active SDE diffusion to a leaf embedding |
| `posterior_from_xt(...)` | Exact posterior for passive diffusion |
| `posterior_from_xt_active(...)` | Exact posterior for active diffusion |
| `belief_propagation(...)` | Full upward + downward BP pass |
| `compute_marginals(...)` | Combine up/down messages into node marginals |
| `run_accuracy_trials(...)` | Average BP accuracy over multiple trials at each time |
