import math
import random

import numpy as np


def index_to_tuple(idx, base, length):
    out = [0] * length
    for pos in range(length - 1, -1, -1):
        out[pos] = idx % base
        idx //= base
    return tuple(out)


def normalize(vec):
    total = sum(vec)
    if total == 0:
        return [1.0 / len(vec) for _ in vec]
    return [x / total for x in vec]


def sample_states_top_down(v, s, L, rules, rng):
    level_sizes = [s ** (L - level) for level in range(L + 1)]
    states = [[0] * level_sizes[level] for level in range(L + 1)]
    states[L][0] = rng.randrange(v)
    for level in range(L, 0, -1):
        for i in range(level_sizes[level]):
            parent_symbol = states[level][i]
            options = rules[level][parent_symbol]
            if not options:
                raise ValueError("No production rules available for a parent symbol.")
            tpl = rng.choice(options)
            base = i * s
            for j in range(s):
                states[level - 1][base + j] = tpl[j]
    return states


def diffuse_forward(x0_index, v, t, T, rng):
    decay = math.exp(-t)
    noise_scale = math.sqrt(T * (1.0 - math.exp(-2.0 * t)))
    x_t = []
    for i in range(v):
        base = decay if i == x0_index else 0.0
        x_t.append(base + noise_scale * rng.normalvariate(0.0, 1.0))
    return x_t


def active_params(k, Ta, tau, t, T):
    if k == 0.0 or tau == 0.0:
        raise ValueError("k and tau must be nonzero for active diffusion.")
    a = math.exp(-k * t)
    b = math.exp(-t / tau)
    c = k + 1.0 / tau
    d = k - 1.0 / tau
    if c == 0.0 or d == 0.0:
        raise ValueError("Invalid active diffusion parameters: c or d is zero.")
    m11 = (T / k) * (1.0 - a * a) + (Ta / (tau * tau)) * (
        (tau / (k * c)) + (1.0 / (d * d)) * (4.0 * a * b / c - b * b * tau - a * a / k)
    )
    m12 = (Ta / (tau * c * d)) * (
        k * (1.0 - b * b) - (1.0 / tau) * (1.0 + b * b - 2.0 * a * b)
    )
    m22 = Ta / tau * (1.0 - b * b)
    alpha = (b - a) / d
    delta = m11 * m22 - m12 * m12
    return a, b, c, d, m11, m12, m22, alpha, delta


def diffuse_forward_active(x0_index, v, t, T, k, Ta, tau, rng):
    a, b, _, _, m11, m12, m22, alpha, _ = active_params(k, Ta, tau, t, T)
    eta0_std = math.sqrt(Ta / tau)
    eta0 = [rng.normalvariate(0.0, eta0_std) for _ in range(v)]
    x_t = []
    eta_t = []
    sqrt_m11 = math.sqrt(max(m11, 0.0))
    if sqrt_m11 == 0.0:
        sqrt_m11 = 1e-12
    var_eta = m22 - (m12 * m12) / m11 if m11 != 0.0 else m22
    if var_eta < 0.0 and abs(var_eta) < 1e-12:
        var_eta = 0.0
    if var_eta < 0.0:
        raise ValueError("Active diffusion covariance is not positive semidefinite.")
    sqrt_var_eta = math.sqrt(max(var_eta, 0.0))
    for i in range(v):
        x0_i = 1.0 if i == x0_index else 0.0
        mean_x = a * x0_i + alpha
        mean_eta = b * eta0[i]
        z1 = rng.normalvariate(0.0, 1.0)
        z2 = rng.normalvariate(0.0, 1.0)
        x_val = mean_x + sqrt_m11 * z1
        eta_val = mean_eta + (m12 / sqrt_m11) * z1 + sqrt_var_eta * z2
        x_t.append(x_val)
        eta_t.append(eta_val)
    return x_t, eta_t


def posterior_from_xt(x_t, t, T):
    denom = T * (1.0 - math.exp(-2.0 * t))
    if denom <= 0.0:
        max_idx = max(range(len(x_t)), key=lambda i: x_t[i])
        out = [0.0] * len(x_t)
        out[max_idx] = 1.0
        return out
    scale = math.exp(-t) / denom
    max_scaled = scale * max(x_t)
    weights = [math.exp(scale * xi - max_scaled) for xi in x_t]
    return normalize(weights)


def posterior_from_xt_active(x_t, eta_t, t, T, k, Ta, tau):
    a, b, _, _, m11, m12, m22, alpha, delta = active_params(k, Ta, tau, t, T)
    denom_inner = (tau / Ta) * delta + m11 * alpha * alpha - 2.0 * m12 * b * alpha + m22 * b * b
    denom = delta * denom_inner
    if denom == 0.0 or delta == 0.0:
        return [1.0 / len(x_t) for _ in x_t]
    term1 = m11 * alpha - m12 * b
    term2 = -m12 * alpha + m22 * b
    exps = []
    max_exp = None
    for x_mu, eta_mu in zip(x_t, eta_t):
        exponent = (
            -term1
            * a
            * (term1 * x_mu + term2 * eta_mu)
            / denom
            + (a / delta) * (m11 * x_mu - m12 * eta_mu)
        )
        if max_exp is None or exponent > max_exp:
            max_exp = exponent
        exps.append(exponent)
    weights = [math.exp(val - max_exp) for val in exps]
    return normalize(weights)


def generate_rules(v, s, m, L, seed=None):
    rng = random.Random(seed)
    total_tuples = v**s
    if m > total_tuples:
        raise ValueError("m cannot exceed v**s")
    if v * m > total_tuples:
        raise ValueError("v * m cannot exceed v**s for disjoint parent rules.")

    rules = {}
    for level in range(1, L + 1):
        level_rules = []
        all_indices = list(range(total_tuples))
        rng.shuffle(all_indices)
        for parent in range(v):
            start = parent * m
            indices = all_indices[start:start + m]
            tuples = [index_to_tuple(idx, v, s) for idx in indices]
            level_rules.append(tuples)
        rules[level] = level_rules
    return rules


def initialize_leaf_messages_uniform(v, leaf_count):
    return [[1.0 / v for _ in range(v)] for _ in range(leaf_count)]


def initialize_leaf_messages_diffusion(
    v, leaf_count, t, T, rules, s, L, seed=None, return_samples=False
):
    rng = random.Random(seed)
    expected_leaf_count = s**L
    if leaf_count != expected_leaf_count:
        raise ValueError("leaf_count does not match s**L for top-down sampling.")
    states = sample_states_top_down(v, s, L, rules, rng)
    leaf_states = states[0]
    messages = []
    x_t_samples = []
    for state in leaf_states:
        x_t = diffuse_forward(state, v, t, T, rng)
        x_t_samples.append(x_t)
        messages.append(posterior_from_xt(x_t, t, T))
    if return_samples:
        return messages, states, x_t_samples
    return messages


def initialize_leaf_messages_active(
    v,
    leaf_count,
    t,
    T,
    k,
    Ta,
    tau,
    rules,
    s,
    L,
    seed=None,
    return_samples=False,
):
    rng = random.Random(seed)
    expected_leaf_count = s**L
    if leaf_count != expected_leaf_count:
        raise ValueError("leaf_count does not match s**L for top-down sampling.")
    states = sample_states_top_down(v, s, L, rules, rng)
    leaf_states = states[0]
    messages = []
    x_t_samples = []
    eta_t_samples = []
    for state in leaf_states:
        x_t, eta_t = diffuse_forward_active(state, v, t, T, k, Ta, tau, rng)
        x_t_samples.append(x_t)
        eta_t_samples.append(eta_t)
        messages.append(posterior_from_xt_active(x_t, eta_t, t, T, k, Ta, tau))
    if return_samples:
        return messages, states, x_t_samples, eta_t_samples
    return messages


def belief_propagation(
    v,
    s,
    L,
    rules,
    leaf_init="diffusion",
    t=1.0,
    T=1.0,
    k=1.0,
    Ta=1.0,
    tau=1.0,
    leaf_seed=None,
    leaf_messages=None,
):
    level_sizes = [s ** (L - level) for level in range(L + 1)]
    up = [[[0.0] * v for _ in range(level_sizes[level])] for level in range(L + 1)]
    down = [[[0.0] * v for _ in range(level_sizes[level])] for level in range(L + 1)]

    # Leaf initialization via diffusion-based posterior or a uniform baseline.
    if leaf_messages is not None:
        up[0] = leaf_messages
    else:
        if leaf_init == "uniform":
            up[0] = initialize_leaf_messages_uniform(v, level_sizes[0])
        elif leaf_init == "diffusion":
            up[0] = initialize_leaf_messages_diffusion(
                v, level_sizes[0], t=t, T=T, rules=rules, s=s, L=L, seed=leaf_seed
            )
        elif leaf_init == "active":
            up[0] = initialize_leaf_messages_active(
                v,
                level_sizes[0],
                t=t,
                T=T,
                k=k,
                Ta=Ta,
                tau=tau,
                rules=rules,
                s=s,
                L=L,
                seed=leaf_seed,
            )
        else:
            raise ValueError(f"Unknown leaf_init: {leaf_init}")

    # Upward pass.
    for level in range(1, L + 1):
        for i in range(level_sizes[level]):
            child_indices = [i * s + j for j in range(s)]
            child_msgs = [up[level - 1][idx] for idx in child_indices]
            unnorm = [0.0] * v
            for y in range(v):
                total = 0.0
                for tpl in rules[level][y]:
                    prod = 1.0
                    for j, x in enumerate(tpl):
                        prod *= child_msgs[j][x]
                    total += prod
                unnorm[y] = total
            up[level][i] = normalize(unnorm)

    # Downward pass: uniform prior at root.
    down[L][0] = [1.0 / v for _ in range(v)]
    for level in range(L, 0, -1):
        for i in range(level_sizes[level]):
            parent_down = down[level][i]
            child_indices = [i * s + j for j in range(s)]
            child_msgs = [up[level - 1][idx] for idx in child_indices]
            for j, child_idx in enumerate(child_indices):
                accum = [0.0] * v
                for y in range(v):
                    y_weight = parent_down[y]
                    if y_weight == 0.0:
                        continue
                    for tpl in rules[level][y]:
                        xj = tpl[j]
                        prod = y_weight
                        for k, x in enumerate(tpl):
                            if k == j:
                                continue
                            prod *= child_msgs[k][x]
                        accum[xj] += prod
                down[level - 1][child_idx] = normalize(accum)

    return up, down


def compute_marginals(up, down):
    marginals = []
    for level_up, level_down in zip(up, down):
        level_marginals = []
        for u, d in zip(level_up, level_down):
            level_marginals.append(normalize([ui * di for ui, di in zip(u, d)]))
        marginals.append(level_marginals)
    return marginals


def argmax_index(vec):
    best_i = 0
    best_v = vec[0]
    for i, val in enumerate(vec[1:], start=1):
        if val > best_v:
            best_v = val
            best_i = i
    return best_i


def accuracy_by_level(marginals, states):
    accuracies = []
    for level, (m_level, s_level) in enumerate(zip(marginals, states)):
        correct = 0
        for i, marg in enumerate(m_level):
            if argmax_index(marg) == s_level[i]:
                correct += 1
        total = len(s_level)
        accuracies.append(correct / total if total else 0.0)
    return accuracies


def logspace(start, end, num):
    if num <= 0:
        return []
    if num == 1:
        return [start]
    log_start = math.log10(start)
    log_end = math.log10(end)
    step = (log_end - log_start) / (num - 1)
    return [10 ** (log_start + i * step) for i in range(num)]


def run_accuracy_trials(
    v,
    s,
    L,
    m,
    t_values,
    T,
    ntrials,
    seed=None,
    leaf_init="diffusion",
    k=1.0,
    Ta=1.0,
    tau=1.0,
):
    rules = generate_rules(v=v, s=s, m=m, L=L, seed=seed)
    rng = random.Random() if seed is None else random.Random(seed)
    acc_by_level = [[] for _ in range(L + 1)]

    for t_index, t in enumerate(t_values, start=1):
        totals = [0.0] * (L + 1)
        for trial in range(1, ntrials + 1):
            print(f"Sim '{leaf_init}': t[{t_index}/{len(t_values)}]={t:.4g}, trial {trial}/{ntrials}")
            trial_seed = None if seed is None else rng.randrange(1 << 30)
            if leaf_init == "diffusion":
                leaf_messages, states, _ = initialize_leaf_messages_diffusion(
                    v, s**L, t, T, rules, s, L, seed=trial_seed, return_samples=True
                )
            elif leaf_init == "active":
                leaf_messages, states, _, _ = initialize_leaf_messages_active(
                    v,
                    s**L,
                    t,
                    T,
                    k,
                    Ta,
                    tau,
                    rules,
                    s,
                    L,
                    seed=trial_seed,
                    return_samples=True,
                )
            elif leaf_init == "uniform":
                states = sample_states_top_down(v, s, L, rules, random.Random(trial_seed))
                leaf_messages = initialize_leaf_messages_uniform(v, s**L)
            else:
                raise ValueError(f"Unknown leaf_init: {leaf_init}")
            up, down = belief_propagation(
                v=v,
                s=s,
                L=L,
                rules=rules,
                leaf_messages=leaf_messages,
                k=k,
                Ta=Ta,
                tau=tau,
            )
            marginals = compute_marginals(up, down)
            acc = accuracy_by_level(marginals, states)
            for i, val in enumerate(acc):
                totals[i] += val
        for i in range(L + 1):
            acc_by_level[i].append(totals[i] / ntrials if ntrials else 0.0)

    return acc_by_level


def plot_accuracy(t_values, acc_by_level):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        for level, acc in enumerate(acc_by_level):
            print(f"Level {level} accuracy:", acc)
        return

    for level, acc in enumerate(acc_by_level):
        plt.plot(t_values, acc, marker="o", label=f"Level {level}")
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel("Accuracy")
    plt.title("BP accuracy vs t")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(t_values, simulations):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available; skipping plot.")
        for sim in simulations:
            print(f"{sim['name']} ({sim['cmap']}):", sim["acc_by_level"])
        return

    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )

    fig, ax = plt.subplots()
    handles = []

    for sim in simulations:
        cmap = plt.get_cmap(sim["cmap"])
        acc_by_level = sim["acc_by_level"]
        level_count = len(acc_by_level)
        colors = cmap(np.linspace(0.15, 0.95, level_count))
        for level, acc in enumerate(acc_by_level):
            ax.plot(t_values, acc, color=colors[level])
        handles.append(
            Line2D([0], [0], color=cmap(0.75), label=f"{sim['name']} ({sim['cmap']})")
        )

    ax.set_xscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Accuracy")
    ax.set_title("BP accuracy vs t")
    ax.legend(handles=handles)
    fig.tight_layout()
    plt.show()


def main():
    v = 32
    s = 2
    L = 10
    m = 8
    seed = None
    T = 1.0
    Tp = 0.001
    Ta = 1.0
    tau = 2.0
    k = 1.0
    ntrials = 100
    t_values = logspace(0.01, 1.0, 40)
    T_activ_equiv = Ta / (k * (1.0 + tau * k))

    acc_diffusion = run_accuracy_trials(
        v=v,
        s=s,
        L=L,
        m=m,
        t_values=t_values,
        T=T,
        ntrials=ntrials,
        seed=seed,
        leaf_init="diffusion",
    )
    acc_active = run_accuracy_trials(
        v=v,
        s=s,
        L=L,
        m=m,
        t_values=t_values,
        T=Tp,
        ntrials=ntrials,
        seed=seed,
        leaf_init="active",
        k=k,
        Ta=Ta,
        tau=tau,
    )
    acc_diffusion_equiv = run_accuracy_trials(
        v=v,
        s=s,
        L=L,
        m=m,
        t_values=t_values,
        T=T_activ_equiv,
        ntrials=ntrials,
        seed=seed,
        leaf_init="diffusion",
    )

    np.savez(
        "bp_accuracy_results.npz",
        t_values=np.array(t_values),
        acc_diffusion=np.array(acc_diffusion, dtype=float),
        acc_active=np.array(acc_active, dtype=float),
        acc_diffusion_equiv=np.array(acc_diffusion_equiv, dtype=float),
        v=v,
        s=s,
        L=L,
        m=m,
        T=T,
        Tp=Tp,
        T_activ_equiv=T_activ_equiv,
        k=k,
        Ta=Ta,
        tau=tau,
        ntrials=ntrials,
    )

    simulations = [
        {"name": "diffusion", "cmap": "viridis", "acc_by_level": acc_diffusion},
        {"name": "active", "cmap": "magma", "acc_by_level": acc_active},
        {"name": "diffusion_equiv", "cmap": "cividis", "acc_by_level": acc_diffusion_equiv},
    ]
    plot_accuracy_comparison(t_values, simulations)


if __name__ == "__main__":
    main()
