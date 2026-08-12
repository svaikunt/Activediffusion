import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from unet_DDPM import Unet


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to broadcast with x_shape"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def _rk45_integrate(
    f,
    y0: torch.Tensor,
    t0: float,
    t1: float,
    h0: float,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    max_step_abs: float | None = None,
    min_step_abs: float | None = None,
    max_steps: int = 20000,
):
    """
    Adaptive Dormand–Prince RK45 (5(4)) ODE solver.
    Integrates dy/dt = f(y, t) from t0 to t1.
    """
    # Dormand–Prince coefficients
    c2 = 1.0 / 5.0
    c3 = 3.0 / 10.0
    c4 = 4.0 / 5.0
    c5 = 8.0 / 9.0
    c6 = 1.0
    c7 = 1.0

    a21 = 1.0 / 5.0

    a31 = 3.0 / 40.0
    a32 = 9.0 / 40.0

    a41 = 44.0 / 45.0
    a42 = -56.0 / 15.0
    a43 = 32.0 / 9.0

    a51 = 19372.0 / 6561.0
    a52 = -25360.0 / 2187.0
    a53 = 64448.0 / 6561.0
    a54 = -212.0 / 729.0

    a61 = 9017.0 / 3168.0
    a62 = -355.0 / 33.0
    a63 = 46732.0 / 5247.0
    a64 = 49.0 / 176.0
    a65 = -5103.0 / 18656.0

    a71 = 35.0 / 384.0
    a72 = 0.0
    a73 = 500.0 / 1113.0
    a74 = 125.0 / 192.0
    a75 = -2187.0 / 6784.0
    a76 = 11.0 / 84.0

    # 5th order solution (same as a7 row)
    b1 = 35.0 / 384.0
    b2 = 0.0
    b3 = 500.0 / 1113.0
    b4 = 125.0 / 192.0
    b5 = -2187.0 / 6784.0
    b6 = 11.0 / 84.0

    # 4th order solution (error estimate)
    b1s = 5179.0 / 57600.0
    b2s = 0.0
    b3s = 7571.0 / 16695.0
    b4s = 393.0 / 640.0
    b5s = -92097.0 / 339200.0
    b6s = 187.0 / 2100.0
    b7s = 1.0 / 40.0

    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    y = y0
    t = float(t0)
    t_end = float(t1)
    direction = 1.0 if t_end > t else -1.0
    h = abs(float(h0)) * direction
    if h == 0.0:
        raise ValueError("RK45 requires a non-zero initial step size h0.")
    if max_step_abs is not None and max_step_abs > 0:
        h = float(min(abs(h), max_step_abs)) * direction

    for _ in range(max_steps):
        if (t - t_end) * direction >= 0:
            break

        # Clamp final step to land exactly on t_end
        if (t + h - t_end) * direction > 0:
            h = (t_end - t)

        k1 = f(y, t)
        k2 = f(y + h * (a21 * k1), t + c2 * h)
        k3 = f(y + h * (a31 * k1 + a32 * k2), t + c3 * h)
        k4 = f(y + h * (a41 * k1 + a42 * k2 + a43 * k3), t + c4 * h)
        k5 = f(y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), t + c5 * h)
        k6 = f(y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), t + c6 * h)

        y5 = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

        k7 = f(
            y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6),
            t + c7 * h,
        )
        y4 = y + h * (b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)

        err = y5 - y4
        scale = atol + rtol * torch.maximum(y.abs(), y5.abs())
        # Use max-norm over all elements for conservative error control in high-dim states.
        err_norm = (err / scale).abs().max().item()

        if err_norm <= 1.0:
            # accept
            t = t + h
            y = y5

        # adapt step size (order 5)
        if err_norm == 0.0:
            factor = max_factor
        else:
            factor = safety * (err_norm ** (-0.2))
            factor = min(max_factor, max(min_factor, factor))
        h = h * factor

        # Clamp adaptive step size to avoid overly large steps (common source of PF failures).
        if max_step_abs is not None and max_step_abs > 0:
            h = float(min(abs(h), max_step_abs)) * direction
        if min_step_abs is not None and min_step_abs > 0:
            if abs(h) < float(min_step_abs):
                h = float(min_step_abs) * direction
        if min_step_abs is not None and min_step_abs > 0 and abs(h) <= 0:
            break

    return y


class CIFAR10Diffusion_SDE(nn.Module):
    """CIFAR-10 passive diffusion matching MNIST structure exactly"""
    
    def __init__(
        self,
        image_size=32,
        in_channels=3,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=64,
        dim_mults=[1, 2, 2, 2],
        num_res_blocks=2,
        T=2.0,
        k=1.0,
        Tp=1.0,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.time_range = T
        self.dt = T / timesteps
        self.k = k
        self.Tp = Tp
        
        self.model = Unet(
            timesteps,
            time_embedding_dim,
            in_channels,
            in_channels,
            base_dim,
            dim_mults,
            num_res_blocks=num_res_blocks,
            dropout=0.1,
            image_size=image_size,
        )

    def _normalize_time(self, t):
        """
        Map continuous time in [0, time_range] to [0, 1] for the embedding layer.
        Without this, all t > 1 collapse to the same embedding, hurting training.
        """
        return torch.clamp(t / self.time_range, min=0.0, max=0.999)

    def _build_pf_time_grid(self, steps, schedule, device, start_time=None):
        """
        Build a descending list of times for the probability-flow ODE integrator.
        Returns a 1-D tensor of length steps+1 (start -> ... -> 0).
        """
        total_start = self.time_range
        start_time = total_start if start_time is None else min(float(start_time), total_start)
        if steps is None or steps <= 0:
            steps = self.timesteps
        steps = max(1, steps)

        t_eps = 1e-3  # match training minimum time; avoid t=0 where model is untrained

        if schedule == "quadratic" and steps > 1:
            # Quadratic striding (CLD-SGM App. E.2.3): concentrate function evaluations
            # near the data end, where the marginal is most complex. Their Tab. 9 shows
            # SSCS 81.1 -> SSCS-QS 20.5 FID at 50 NFE.
            idx = torch.linspace(0, 1, steps + 1, device=device)
            asc = t_eps + (start_time - t_eps) * idx ** 2
            return torch.flip(asc, dims=[0])

        if schedule == "log" and steps > 1:
            min_time = max(self.dt, t_eps)
            if start_time <= min_time:
                schedule = "linear"
            else:
                idx = torch.linspace(0, 1, steps + 1, device=device)
                asc = torch.exp(
                    torch.log(torch.tensor(min_time, device=device))
                    + idx * torch.log(torch.tensor(start_time / min_time, device=device))
                )
                times = torch.flip(asc, dims=[0])
                return times

        return torch.linspace(start_time, t_eps, steps + 1, device=device)

    def marginal_prob(self, x_0, t):
        """
        Returns mean and standard deviation of the marginal distribution p(x_t|x_0)
        for the basic diffusion SDE: dx = - x * dt + dw
        EXACTLY like MNIST implementation
        """
        mean = x_0 * torch.exp(-self.k * t)[:, None, None, None]
        std = torch.sqrt((self.Tp / self.k) * (1 - torch.exp(-2 * self.k * t)))[:, None, None, None]
        
        return mean, std

    def forward(self, x, noise):
        """Forward pass - EXACTLY like MNIST"""
        # Generate random time between 0 and 2
        t = 1e-3 + (self.time_range - 1e-3) * torch.rand(x.shape[0], device=x.device)
        
        # Get noisy image and score
        mean, std = self.marginal_prob(x, t)
        x_t = mean + std * noise
        score = self.model(x_t, self._normalize_time(t))
        
        # Return for loss computation
        return x_t, score, mean, std
    
    def compute_loss(self, x, noise=None):
        """Compute loss with std^2 weighting (not std^4) for better fine detail learning"""
        if noise is None:
            noise = torch.randn_like(x)
        
        x_t, score, mean, std = self.forward(x, noise)
        true_score = -(x_t - mean)/(std**2)
        # Weight by std^2 (not std^4) to match Song's framework: λ(t) ∝ σ(t)²
        loss = torch.nn.functional.mse_loss(std * score, std * true_score)
        return loss
    
    @torch.no_grad()
    def sampling(self, n_samples, device="cuda", probability_flow=False, pf_steps=None, pf_schedule="linear", pf_solver="heun", pf_rtol=1e-4, pf_atol=1e-5, tweedie=True):
        """
        Generate samples using Euler-Maruyama solver for the reverse-time SDE
        EXACTLY like MNIST implementation
        """
        # Initialize from stationary distribution N(0, Tp/k)
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        x_t = x_t * math.sqrt(self.Tp / self.k)

        if probability_flow and pf_solver == "rk45":
            t0 = float(self.time_range)
            t1 = 0.0
            steps = self.timesteps if (pf_steps is None or pf_steps <= 0) else int(pf_steps)
            h0 = (t0 - t1) / float(max(1, steps))
            # Cap max step to the nominal fixed-step size for stability.
            max_step_abs = abs(h0)

            def drift_fn(x, t_scalar: float):
                t_tensor = torch.full((n_samples,), float(t_scalar), device=device)
                score = self.model(x, self._normalize_time(t_tensor))
                return -self.k * x - self.Tp * score

            x_t = _rk45_integrate(
                drift_fn,
                x_t,
                t0=t0,
                t1=t1,
                h0=h0,
                rtol=pf_rtol,
                atol=pf_atol,
                max_step_abs=max_step_abs,
            )
        elif probability_flow:
            # Heun-Kutta (HK) integration for the PF-ODE: dx/dt = -k*x - Tp*score
            time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, x_t.device)
            for idx in tqdm(range(len(time_grid) - 1), desc="Sampling (PF-ODE, HK)"):
                t_curr = time_grid[idx]
                t_next = time_grid[idx + 1]
                dt = (t_curr - t_next).clamp(min=1e-6)

                # First stage (Euler predictor at t_curr)
                t_curr_tensor = torch.full((n_samples,), t_curr.item(), device=device)
                score_curr = self.model(x_t, self._normalize_time(t_curr_tensor))
                drift_curr = -self.k * x_t - self.Tp * score_curr

                x_pred = x_t - drift_curr * dt  # Euler prediction to t_next

                # Second stage (corrector using drift at t_next)
                t_next_tensor = torch.full((n_samples,), t_next.item(), device=device)
                score_next = self.model(x_pred, self._normalize_time(t_next_tensor))
                drift_next = -self.k * x_pred - self.Tp * score_next

                # Heun update (average of drifts)
                x_t = x_t - 0.5 * (drift_curr + drift_next) * dt

            if tweedie:
                t_last = time_grid[-1].item()
                t_last_tensor = torch.full((n_samples,), t_last, device=x_t.device)
                score_last = self.model(x_t, self._normalize_time(t_last_tensor))
                a_last = math.exp(-self.k * t_last)
                std2_last = (self.Tp / self.k) * (1.0 - math.exp(-2.0 * self.k * t_last))
                x_t = (x_t + std2_last * score_last) / a_last
        else:
            # SDE (Euler-Maruyama) sampler with a configurable step count/schedule
            # (reuses the same pf_steps/pf_schedule knobs as the PF-ODE branch; at the
            # default pf_steps=None this is effectively the old fixed-self.dt loop,
            # off by a single negligible step at the t=T boundary).
            time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, x_t.device)
            n_intervals = len(time_grid) - 1
            for idx in tqdm(range(n_intervals), desc="Sampling"):
                t_curr = time_grid[idx]
                t_next = time_grid[idx + 1]
                dt = (t_curr - t_next).clamp(min=1e-6)
                is_final_step = tweedie and (idx == n_intervals - 1)

                t_tensor = torch.full((n_samples,), t_curr.item(), device=device)
                score = self.model(x_t, self._normalize_time(t_tensor))

                drift = -self.k * x_t - 2 * self.Tp * score
                x_t = x_t - drift * dt

                if not is_final_step:
                    noise_scale = math.sqrt(2 * self.Tp * dt.item())
                    x_t = x_t + noise_scale * torch.randn_like(x_t)

            if tweedie:
                t_last = time_grid[-1].item()
                t_last_tensor = torch.full((n_samples,), t_last, device=x_t.device)
                score_last = self.model(x_t, self._normalize_time(t_last_tensor))
                a_last = math.exp(-self.k * t_last)
                std2_last = (self.Tp / self.k) * (1.0 - math.exp(-2.0 * self.k * t_last))
                x_t = (x_t + std2_last * score_last) / a_last

        # Scale to [0, 1] range
        x_t = x_t.clip(-1, 1)
        x_t = (x_t + 1.) / 2.
        return x_t

    @torch.no_grad()
    def sampling_from_intermediate(self, x_t, t_start, device="cuda", probability_flow=False, pf_steps=None, pf_schedule="linear", pf_solver="heun", pf_rtol=1e-4, pf_atol=1e-5):
        """
        Generate samples starting from an intermediate noisy state x_t at time t_start
        EXACTLY like MNIST implementation
        """
        # Initialize from provided state
        batch_size = x_t.shape[0]
        
        # Adjust t_start to be consistent with the regular sampling function
        adjusted_t_start = min(t_start, self.timesteps - 2)
        
        if probability_flow and pf_solver == "rk45":
            t0 = float(adjusted_t_start * self.dt)
            t1 = 0.0
            steps = self.timesteps if (pf_steps is None or pf_steps <= 0) else int(pf_steps)
            h0 = (t0 - t1) / float(max(1, steps))
            max_step_abs = abs(h0)

            def drift_fn(x, t_scalar: float):
                t_tensor = torch.full((batch_size,), float(t_scalar), device=device)
                score = self.model(x, self._normalize_time(t_tensor))
                return -self.k * x - self.Tp * score

            x_t = _rk45_integrate(
                drift_fn,
                x_t,
                t0=t0,
                t1=t1,
                h0=h0,
                rtol=pf_rtol,
                atol=pf_atol,
                max_step_abs=max_step_abs,
            )
        elif probability_flow:
            # Heun-Kutta (HK) integration for PF-ODE from intermediate state
            start_time = adjusted_t_start * self.dt
            time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, x_t.device, start_time=start_time)
            for idx in tqdm(range(len(time_grid) - 1), desc="Sampling from intermediate (PF-ODE, HK)"):
                t_curr = time_grid[idx]
                t_next = time_grid[idx + 1]
                dt = (t_curr - t_next).clamp(min=1e-6)

                # First stage (Euler predictor at t_curr)
                t_curr_tensor = torch.full((batch_size,), t_curr.item(), device=device)
                score_curr = self.model(x_t, self._normalize_time(t_curr_tensor))
                drift_curr = -self.k * x_t - self.Tp * score_curr

                x_pred = x_t - drift_curr * dt  # Euler prediction to t_next

                # Second stage (corrector using drift at t_next)
                t_next_tensor = torch.full((batch_size,), t_next.item(), device=device)
                score_next = self.model(x_pred, self._normalize_time(t_next_tensor))
                drift_next = -self.k * x_pred - self.Tp * score_next

                # Heun update (average of drifts)
                x_t = x_t - 0.5 * (drift_curr + drift_next) * dt
        else:
            noise_scale = math.sqrt(2 * self.Tp * self.dt)
            for i in tqdm(range(adjusted_t_start, -1, -1), desc="Sampling from intermediate"):
                t = torch.full((batch_size,), i * self.dt, device=device)
                score = self.model(x_t, self._normalize_time(t))
                drift = -self.k * x_t - 2 * self.Tp * score
                x_t = x_t - drift * self.dt

                # Add noise for all steps except the final one (i=0)
                if i > 0:
                    x_t = x_t + noise_scale * torch.randn_like(x_t)
        
        # Scale to [0, 1] range
        x_t = x_t.clip(-1, 1)
        x_t = (x_t + 1.) / 2.
        return x_t


class CIFAR10_Active_Diffusion_SDE(nn.Module):
    """CIFAR-10 active diffusion using 3 independent MNIST-style systems"""
    
    def __init__(
        self,
        image_size=32,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=64,
        dim_mults=[1, 2, 2, 2],
        num_res_blocks=2,
        Tp=1e-3,
        Ta=1.0,
        k=1.0,
        tau=0.4,
        T=2.0,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = 6  # RGB + 3 eta channels
        self.image_size = image_size
        self.dt = T/timesteps  # Use T parameter like MNIST
        self.time_range = T
        
        # Physical parameters - EXACTLY like MNIST
        self.Tp = Tp
        self.Ta = Ta
        self.k = k
        self.tau = tau
        self.T = T
        eta_scale = math.sqrt(self.Ta / self.tau)
        self.register_buffer("eta_scale_tensor", torch.tensor(eta_scale, dtype=torch.float32))
        
        # Model outputs 6 channels for F_R, F_η_r, F_G, F_η_g, F_B, F_η_b
        self.model = Unet(
            timesteps,
            time_embedding_dim,
            6,
            6,
            base_dim,
            dim_mults,
            num_res_blocks=num_res_blocks,
            dropout=0.1,
            image_size=image_size,
        )

    def _normalize_time(self, t):
        """Normalize continuous time to [0, 1] for the embedding layer."""
        return torch.clamp(t / self.time_range, min=0.0, max=0.999)

    def _build_pf_time_grid(self, steps, schedule, device, start_time=None, t_end=None):
        """`t_end` sets where the grid stops (default 1e-3, the training minimum time).

        For the ACTIVE model, t_end=0.0 is a reasonable choice: with Tp ~ 1e-3 the
        x-channel's residual noise at t=1e-3 is only sigma ~ 0.0015 (vs ~0.045 for the
        passive model at Tp=1.0, a 29x difference), so there is essentially nothing left
        to denoise and the sampler can integrate straight to zero. Note the network is
        still never *evaluated* below the second-to-last grid point, since the score is
        read at the start (or midpoint) of each interval, not at its endpoint.
        """
        total_start = self.time_range
        start_time = total_start if start_time is None else min(float(start_time), total_start)
        if steps is None or steps <= 0:
            steps = self.timesteps
        steps = max(1, steps)

        t_eps = 1e-3 if t_end is None else float(t_end)

        if schedule == "quadratic" and steps > 1:
            # Quadratic striding (CLD-SGM App. E.2.3): concentrate function evaluations
            # near the data end, where the marginal is most complex. Their Tab. 9 shows
            # SSCS 81.1 -> SSCS-QS 20.5 FID at 50 NFE.
            idx = torch.linspace(0, 1, steps + 1, device=device)
            asc = t_eps + (start_time - t_eps) * idx ** 2
            return torch.flip(asc, dims=[0])

        if schedule == "log" and steps > 1 and t_eps > 0:
            min_time = max(self.dt, t_eps)
            if start_time <= min_time:
                schedule = "linear"
            else:
                idx = torch.linspace(0, 1, steps + 1, device=device)
                asc = torch.exp(
                    torch.log(torch.tensor(min_time, device=device))
                    + idx * torch.log(torch.tensor(start_time / min_time, device=device))
                )
                times = torch.flip(asc, dims=[0])
                return times

        return torch.linspace(start_time, t_eps, steps + 1, device=device)

    def generate_eta0(self, batch_size, device=None):
        """Generate eta_0 for all 3 RGB channels - EXACTLY like MNIST"""
        variance = self.Ta / self.tau
        target_device = device if device is not None else torch.device("cpu")
        std = torch.sqrt(torch.tensor(variance, device=target_device, dtype=torch.float32))
        # Generate 3 independent eta fields: [batch, 3, H, W]
        return torch.randn(batch_size, 3, self.image_size, self.image_size, device=target_device) * std

    def compute_covariance(self, t):
        """Compute covariance matrix - EXACTLY like original MNIST"""
        batch_size = t.shape[0]
        Tp = torch.full((batch_size,), self.Tp, device=t.device)
        Ta = torch.full((batch_size,), self.Ta, device=t.device)
        k = torch.full((batch_size,), self.k, device=t.device)
        tau = torch.full((batch_size,), self.tau, device=t.device)
        
        # Compute intermediate terms (original formulation)
        a = torch.exp(-k * t)
        b = torch.exp(-t / tau)
        Tx = Tp
        Ty = Ta / (tau * tau)
        w = 1 / tau
        
        # Compute matrix elements with proper active matter physics
        M11 = (1/k)*Tx*(1-a*a) + (1/k)*Ty*(
            1/(w*(k+w)) + 
            4*a*b*k/((k+w)*(k-w)**2) - 
            (k*b*b + w*a*a)/(w*(k-w)**2) + 1e-8
        )
        M12 = (Ty/(w*(k*k - w*w))) * (k*(1-b*b) - w*(1 + b*b - 2*a*b))
        M22 = (Ty/w)*(1-b*b) + 1e-8
        
        # Stack elements to form covariance matrix
        cov_matrix = torch.stack([
            torch.stack([M11, M12], dim=1),
            torch.stack([M12, M22], dim=1)
        ], dim=1)
        
        return cov_matrix

    def generate_correlated_noise(self, covariance_matrix, num_channels=3):
        """Generate correlated noise for all RGB channels"""
        batch_size = covariance_matrix.shape[0]
        device = covariance_matrix.device
        covariance_matrix = covariance_matrix.view(batch_size, 2, 2)
        
        # Try automatic Cholesky first
        try:
            L = torch.linalg.cholesky(covariance_matrix)
        except RuntimeError as e:
            if "positive-definite" in str(e):
                # Add small regularization
                reg = 1e-6 * torch.eye(2, device=device).unsqueeze(0).expand_as(covariance_matrix)
                L = torch.linalg.cholesky(covariance_matrix + reg)
            else:
                raise e
        
        # Generate 3 independent correlated noise pairs: (R,η_r), (G,η_g), (B,η_b)
        all_noise = []
        for channel in range(num_channels):
            # Generate uncorrelated noise for this channel
            uncorrelated_noise = torch.randn(batch_size, 2, self.image_size, self.image_size, device=device)
            
            # Reshape for batch matrix multiplication
            uncorrelated_noise_flat = uncorrelated_noise.view(batch_size, 2, -1)
            
            # Apply Cholesky factor to get correlated noise
            correlated_noise_flat = torch.bmm(L, uncorrelated_noise_flat)
            
            # Reshape back to image format
            correlated_noise = correlated_noise_flat.view(batch_size, 2, self.image_size, self.image_size)
            all_noise.append(correlated_noise)
        
        # Stack all channel noise: [batch, 6, H, W] = [R,η_r,G,η_g,B,η_b]
        return torch.cat(all_noise, dim=1)

    def compute_mean(self, x_0, eta_0, t):
        """Compute mean values at time t - EXACTLY like MNIST"""
        k = self.k
        tau = self.tau
        
        # Reshape t for broadcasting
        t_view = t.view(-1, 1, 1, 1)
        
        # Compute coefficients (original formulation)
        a = torch.exp(-k * t_view)
        b = (torch.exp(-t_view / tau) - torch.exp(-k * t_view)) / (k - (1 / tau))
        c = torch.exp(-t_view / tau)
        
        # Compute means for all channels
        mean_x = a * x_0 + b * eta_0
        mean_eta = c * eta_0
        
        return mean_x, mean_eta

    def marginal_prob(self, x_0, eta_0, t):
        """Returns mean and covariance matrix - EXACTLY like MNIST"""
        # Compute means
        mean_x, mean_eta = self.compute_mean(x_0, eta_0, t)
        
        # Compute covariance matrix
        cov = self.compute_covariance(t)
        
        return (mean_x, mean_eta), cov

    def forward(self, x, t=None, noise=None):
        """Forward pass - adapted MNIST structure for RGB"""
        # Split input channels
        batch_size = x.shape[0]
        x_0 = x[:, :3]   # RGB channels
        eta_0 = x[:, 3:]  # 3 eta channels
        
        # Generate random time if not provided - EXACTLY like MNIST
        if t is None:
            t = 1e-3 + (self.T-1e-3)*torch.rand(batch_size, device=x.device)
        
        # Get means and covariance
        (mean_x, mean_eta), cov = self.marginal_prob(x_0, eta_0, t)
        
        # Generate correlated noise if not provided
        if noise is None:
            noise = self.generate_correlated_noise(cov, num_channels=3)
        
        # Extract noise components for all channels
        noise_rgb = noise[:, [0,2,4]]  # R, G, B noise
        noise_eta = noise[:, [1,3,5]]  # η_r, η_g, η_b noise
        
        # Compute noisy samples
        x_t = mean_x + noise_rgb
        eta_t = mean_eta + noise_eta
        
        # Interleave for model input: [R,η_r,G,η_g,B,η_b]
        scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x.device)
        model_input = torch.zeros(batch_size, 6, self.image_size, self.image_size, device=x.device)
        model_input[:, [0,2,4]] = x_t    # R, G, B
        model_input[:, [1,3,5]] = eta_t / scale  # scaled η channels
        
        # Get forces from model
        model_output = self.model(model_input, self._normalize_time(t))
        F_x = model_output[:, [0,2,4]]   # F_R, F_G, F_B
        F_eta = model_output[:, [1,3,5]] # F_η_r, F_η_g, F_η_b
        
        return (x_t, eta_t), (F_x, F_eta), (mean_x, mean_eta), cov, noise

    def diffusion_loss_active(self, model_output, rgb_images, eta_0, t, noise=None):
        """Compute loss - EXACTLY like MNIST but for 3 channels"""
        device = rgb_images.device
        t = t.view(-1, 1, 1, 1)
        
        # Split model output
        F_x, F_eta = model_output
        
        # Compute mean
        mean_x, mean_eta = self.compute_mean(rgb_images, eta_0, t)
        
        # Compute covariance matrix
        M = self.compute_covariance(t.squeeze())
        
        # Use same noise from forward pass if provided
        if noise is None:
            noise = self.generate_correlated_noise(M, num_channels=3)
        
        # Extract noise components
        noise_rgb = noise[:, [0,2,4]]
        noise_eta = noise[:, [1,3,5]]
        
        # Compute noisy samples using the same noise
        xt = mean_x + noise_rgb
        etat = mean_eta + noise_eta
        
        # Extract covariance matrix components
        M11 = M[:, 0, 0].view(-1, 1, 1, 1)
        M12 = M[:, 0, 1].view(-1, 1, 1, 1)
        M22 = M[:, 1, 1].view(-1, 1, 1, 1)
        det = M11 * M22 - M12 * M12
        
        # Protect determinant from becoming too small (critical for numerical stability)
        det = torch.clamp(det, min=1e-8)
        
        # Compute mean coefficients
        k = self.k
        tau = self.tau
        a = torch.exp(-k * t)
        b = (torch.exp(-t / tau) - torch.exp(-k * t)) / (k - (1 / tau))
        c = torch.exp(-t / tau)
        
        # Apply MNIST loss to each channel independently
        total_loss = 0.0
        for channel in range(3):  # R, G, B
            # Extract channel-specific values
            rgb_ch = xt[:, channel:channel+1]
            eta_ch = etat[:, channel:channel+1] 
            rgb_orig_ch = rgb_images[:, channel:channel+1]
            eta_0_ch = eta_0[:, channel:channel+1]
            F_rgb_ch = F_x[:, channel:channel+1]
            F_eta_ch = F_eta[:, channel:channel+1]
            
            # Compute loss terms (ORIGINAL MNIST formulation)
            Feta = torch.sqrt(1 / det) * (-M11 * (eta_ch - c * eta_0_ch) + M12 * (rgb_ch - a * rgb_orig_ch - b * eta_0_ch))
            Fx = torch.sqrt(1 / det) * (-M22 * (rgb_ch - a * rgb_orig_ch - b * eta_0_ch) + M12 * (eta_ch - c * eta_0_ch))
            
            scr_eta = torch.sqrt(det) * F_eta_ch
            scr_x = torch.sqrt(det) * F_rgb_ch
            
            # Compute loss for this channel
            loss_eta_ch = torch.mean((scr_eta - Feta) ** 2)
            loss_x_ch = torch.mean((scr_x - Fx) ** 2)
            
            # Add to total
            total_loss += self.Ta * loss_eta_ch + self.Tp * loss_x_ch
        
        # Average across channels to maintain active ≈ passive property
        total_loss = total_loss / 3.0
        
        return total_loss

    def _active_pf_drift(self, x_eta, model_output, Tp_tensor, Ta_tensor, tau_tensor):
        """Compute deterministic probability-flow drift for active sampler."""
        drift = torch.zeros_like(x_eta)
        force_coeff_rgb = Tp_tensor
        force_coeff_eta = Ta_tensor / (tau_tensor * tau_tensor)

        for channel in range(3):
            rgb_idx = channel * 2
            eta_idx = channel * 2 + 1

            rgb = x_eta[:, rgb_idx:rgb_idx+1]
            eta = x_eta[:, eta_idx:eta_idx+1]
            F_rgb = model_output[:, rgb_idx:rgb_idx+1]
            F_eta = model_output[:, eta_idx:eta_idx+1]

            drift[:, rgb_idx:rgb_idx+1] = (self.k * rgb - eta) + force_coeff_rgb * F_rgb
            drift[:, eta_idx:eta_idx+1] = (eta / tau_tensor) + force_coeff_eta * F_eta

        return drift

    def _active_tweedie_correction(self, x_eta, t_eps, device):
        """Exact final-step correction at t_eps: multivariate Tweedie/Miyasawa estimate
        of x_0 given the joint (x_t, eta_t) state, using the network's full (F_x, F_eta)
        score output and the exact forward transition coefficients (a, b, c) and
        covariance (M11, M12, M22) at t_eps -- not the small-t_eps linearization used
        in an earlier draft (which dropped a same-order k*t_eps*x_t term; see
        active_sscs_sampler_note.tex for the derivation and the numerical check that
        exposed the missing term). Costs one extra network evaluation, matching the
        passive model's Tweedie step; F_eta is free since it comes from the same call."""
        n_samples = x_eta.shape[0]
        scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(device)
        t_tensor = torch.full((n_samples,), t_eps, device=device)
        model_input = x_eta.clone()
        model_input[:, [1, 3, 5]] = model_input[:, [1, 3, 5]] / scale
        model_output = self.model(model_input, self._normalize_time(t_tensor))

        x_t = x_eta[:, [0, 2, 4]]
        eta_t = x_eta[:, [1, 3, 5]]
        F_x = model_output[:, [0, 2, 4]]
        F_eta = model_output[:, [1, 3, 5]]

        k = self.k
        tau = self.tau
        a = math.exp(-k * t_eps)
        c = math.exp(-t_eps / tau)
        b = (c - a) / (k - 1.0 / tau)

        cov = self.compute_covariance(t_tensor)
        M11 = cov[:, 0, 0].view(-1, 1, 1, 1)
        M12 = cov[:, 0, 1].view(-1, 1, 1, 1)
        M22 = cov[:, 1, 1].view(-1, 1, 1, 1)

        x0_hat = (1.0 / a) * (
            (x_t + M11 * F_x + M12 * F_eta) - (b / c) * (eta_t + M12 * F_x + M22 * F_eta)
        )

        corrected = x_eta.clone()
        corrected[:, [0, 2, 4]] = x0_hat
        return corrected

    def _stationary_covariance(self):
        """Stationary covariance of the forward (x, eta) OU process: the solution of
        A S + S A^T + Q = 0 for A = [[-k, 1], [0, -1/tau]], Q = diag(2Tp, 2Ta/tau^2).
        Returns (sxx, sxe, see) as plain floats."""
        see = self.Ta / self.tau
        sxe = see / (self.k + 1.0 / self.tau)
        sxx = sxe / self.k + self.Tp / self.k
        return sxx, sxe, see

    def _eq_transition(self, s):
        """Transition matrix E = exp(L' s) and covariance C for the EQUILIBRIUM-corrected
        linear flow, where L' = -A - Q S_inf^{-1} = S_inf A^T S_inf^{-1}.

        Unlike the naive split's L = -A (which is *unstable*, eigenvalues +k, +1/tau),
        L' is similar to A^T and therefore stable (eigenvalues -k, -1/tau), with
        stationary covariance exactly S_inf. This is the split CLD-SGM's SSCS actually
        uses -- their `score + m_inv*v` is the same equilibrium subtraction. Keeping the
        equilibrium part in the *analytic* flow leaves only the small residual score for
        the Euler substep, instead of two huge nearly-cancelling operations.

        For a stable OU with stationary S: cov over s is  S - E S E^T.
        """
        k, tau = self.k, self.tau
        sxx, sxe, see = self._stationary_covariance()
        a = math.exp(-k * s)
        c = math.exp(-s / tau)
        b = (c - a) / (k - 1.0 / tau)

        # P = S_inf @ exp(A^T s), with exp(A^T s) = [[a, 0], [b, c]]
        p11 = sxx * a + sxe * b
        p12 = sxe * c
        p21 = sxe * a + see * b
        p22 = see * c

        d = sxx * see - sxe * sxe
        e11 = (p11 * see - p12 * sxe) / d
        e12 = (-p11 * sxe + p12 * sxx) / d
        e21 = (p21 * see - p22 * sxe) / d
        e22 = (-p21 * sxe + p22 * sxx) / d

        es11 = e11 * sxx + e12 * sxe
        es12 = e11 * sxe + e12 * see
        es21 = e21 * sxx + e22 * sxe
        es22 = e21 * sxe + e22 * see
        c11 = sxx - (es11 * e11 + es12 * e12)
        c12 = sxe - (es11 * e21 + es12 * e22)
        c22 = see - (es21 * e21 + es22 * e22)
        return (e11, e12, e21, e22), (c11, c12, c22)

    def _eq_half_step(self, x_eta, s, add_noise=True):
        """Exact propagation of the equilibrium-corrected linear flow over length s."""
        (e11, e12, e21, e22), (c11, c12, c22) = self._eq_transition(float(s))
        x = x_eta[:, [0, 2, 4]]
        e = x_eta[:, [1, 3, 5]]
        out = torch.zeros_like(x_eta)
        out[:, [0, 2, 4]] = e11 * x + e12 * e
        out[:, [1, 3, 5]] = e21 * x + e22 * e
        if not add_noise:
            return out
        cov = torch.tensor([[c11, c12], [c12, c22]], device=x_eta.device, dtype=torch.float32)
        cov = cov.unsqueeze(0).expand(x_eta.shape[0], 2, 2).contiguous()
        return out + self.generate_correlated_noise(cov, num_channels=3)

    def _eq_score_force(self, model_output, x_eta):
        """Q * (score - equilibrium score): the residual force for the equilibrium split.
        The equilibrium score -S_inf^{-1} z is already carried by _eq_half_step, so only
        the residual (which vanishes as t -> T) goes through the Euler substep."""
        sxx, sxe, see = self._stationary_covariance()
        d = sxx * see - sxe * sxe
        i00, i01, i11 = see / d, -sxe / d, sxx / d
        x = x_eta[:, [0, 2, 4]]
        e = x_eta[:, [1, 3, 5]]
        force = torch.zeros_like(x_eta)
        force[:, [0, 2, 4]] = 2.0 * self.Tp * (model_output[:, [0, 2, 4]] + (i00 * x + i01 * e))
        force[:, [1, 3, 5]] = (2.0 * self.Ta / (self.tau ** 2)) * (
            model_output[:, [1, 3, 5]] + (i01 * x + i11 * e)
        )
        return force

    def _score_force(self, model_output, Tp_tensor, Ta_tensor, tau_tensor):
        """Score-network force term only (the nonlinear 'N' part of the split);
        excludes the linear (kx-eta, eta/tau) drift, which the exact half-step handles."""
        force = torch.zeros_like(model_output)
        force[:, [0, 2, 4]] = 2 * Tp_tensor * model_output[:, [0, 2, 4]]
        force[:, [1, 3, 5]] = (2 * Ta_tensor / (tau_tensor * tau_tensor)) * model_output[:, [1, 3, 5]]
        return force

    def _reverse_transition_mean(self, x0, eta0, s):
        """Exact mean of the reverse-time linear (L) flow over an interval of length s.
        See active_sscs_sampler_note.tex, Eq. 6."""
        k = self.k
        tau = self.tau
        e1 = math.exp(k * s)
        e2 = math.exp(s / tau)
        delta = k - 1.0 / tau
        mean_x = e1 * x0 + ((e2 - e1) / delta) * eta0
        mean_eta = e2 * eta0
        return mean_x, mean_eta

    def _reverse_transition_covariance(self, s, batch_size, device):
        """Exact noise covariance of the reverse-time linear (L) flow over an interval
        of length s, shape [batch_size, 2, 2]. See active_sscs_sampler_note.tex, Eqs. 7-9."""
        k = self.k
        tau = self.tau
        q1 = 2.0 * self.Tp
        q2 = 2.0 * self.Ta / (tau * tau)
        delta = k - 1.0 / tau
        mu = k + 1.0 / tau

        e1 = math.exp(k * s)
        e2 = math.exp(s / tau)

        sigma_etaeta = (self.Ta / tau) * (e2 * e2 - 1.0)
        sigma_xeta = (q2 / delta) * ((tau / 2.0) * (e2 * e2 - 1.0) - (e1 * e2 - 1.0) / mu)
        sigma_xx = (q1 * (e1 * e1 - 1.0)) / (2.0 * k) + (q2 / (delta * delta)) * (
            (tau / 2.0) * (e2 * e2 - 1.0) - 2.0 * (e1 * e2 - 1.0) / mu + (e1 * e1 - 1.0) / (2.0 * k)
        )

        cov = torch.tensor(
            [[sigma_xx, sigma_xeta], [sigma_xeta, sigma_etaeta]], device=device, dtype=torch.float32
        )
        return cov.unsqueeze(0).expand(batch_size, 2, 2).contiguous()

    def _analytic_half_step(self, x_eta, s, add_noise=True):
        """Exact propagation of the reverse-time linear (L) part over an interval of
        length s: (x, eta) -> exact OU mean, plus exact covariance noise if add_noise."""
        batch_size = x_eta.shape[0]
        device = x_eta.device

        rgb = x_eta[:, [0, 2, 4]]
        eta = x_eta[:, [1, 3, 5]]
        mean_rgb, mean_eta = self._reverse_transition_mean(rgb, eta, s)

        mean_x_eta = torch.zeros_like(x_eta)
        mean_x_eta[:, [0, 2, 4]] = mean_rgb
        mean_x_eta[:, [1, 3, 5]] = mean_eta

        if not add_noise:
            return mean_x_eta

        cov = self._reverse_transition_covariance(s, batch_size, device)
        noise = self.generate_correlated_noise(cov, num_channels=3)
        return mean_x_eta + noise

    @torch.no_grad()
    def sampling_sscs(self, n_samples, device="cuda", tweedie=True, pf_steps=None, pf_schedule="linear", score_time="midpoint", splitting="equilibrium", t_end=None):
        """Symmetric-splitting (SSCS-style) stochastic sampler for the active SDE.

        Strang split: analytic linear half-step, one Euler score substep, analytic
        linear half-step -- one network evaluation per step, same cost as EM.

        `splitting` selects WHERE the split is made, which turns out to matter enormously:

          "equilibrium" (default, and what CLD-SGM's SSCS actually does): the linear
            flow absorbs the equilibrium score -S_inf^{-1} z, giving the *stable* generator
            L' = S_inf A^T S_inf^{-1} (eigenvalues -k, -1/tau) whose stationary covariance
            is exactly S_inf. The Euler substep then only carries the small residual
            score, which vanishes as t -> T.

          "naive": the linear flow is just -A (eigenvalues +k, +1/tau -- UNSTABLE) and the
            Euler substep carries the full score. The two are huge and nearly cancel, so
            the Euler substep's O(h^2) error rides on an enormous constant and the sampler
            loses variance badly.

        Measured against an exact-Gaussian-score benchmark (var error vs the analytic
        target, x / eta channels):
            50 steps : equilibrium -1.4% / -2.0% | naive -22.7% / -18.1% | EM -1.5% / +10.8%
            500 steps: equilibrium -0.3% / -0.3% | naive  -3.1% /  -2.3% | EM -0.3% /  +1.1%
        i.e. the equilibrium split beats EM everywhere (5x better on eta at 50 steps),
        while the naive split is far worse than the EM it was meant to improve on.

        `t_end`: where the time grid stops (default 1e-3). Pass 0.0 to integrate straight
        to t=0 -- reasonable for the active model since Tp ~ 1e-3 leaves almost no residual
        x-noise. When t_end <= 0 the Tweedie correction is skipped (the state is already
        at t=0 and the correction's coefficients are singular there).

        `score_time`: "midpoint" (default) scores at t_curr - half_dt, matching the state
        after the first half-step; "start" reproduces the older stale-label behavior.
        """
        if t_end is not None and float(t_end) <= 0.0:
            tweedie = False

        Tp_tensor = torch.tensor(self.Tp, device=device)
        Ta_tensor = torch.tensor(self.Ta, device=device)
        tau_tensor = torch.tensor(self.tau, device=device)
        scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(device)

        use_eq = (splitting == "equilibrium")
        if use_eq:
            # Initialize from the true stationary covariance S_inf, which is also the
            # equilibrium-flow's invariant distribution. (The naive path's M(T) init
            # understates var_x by ~4% at T=2, k=1.)
            sxx, sxe, see = self._stationary_covariance()
            cov0 = torch.tensor([[sxx, sxe], [sxe, see]], device=device, dtype=torch.float32)
            cov0 = cov0.unsqueeze(0).expand(n_samples, 2, 2).contiguous()
            x_eta = self.generate_correlated_noise(cov0, num_channels=3)
        else:
            base_cov = self.compute_covariance(self.T * torch.ones(n_samples, device=device))
            x_eta = self.generate_correlated_noise(base_cov, num_channels=3)

        time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, device, t_end=t_end)
        n_intervals = len(time_grid) - 1
        for idx in tqdm(range(n_intervals), desc="Sampling (SSCS)"):
            t_curr = time_grid[idx]
            t_next = time_grid[idx + 1]
            dt = (t_curr - t_next).clamp(min=1e-6)
            half_dt = dt / 2.0
            is_final_step = tweedie and (idx == n_intervals - 1)

            if use_eq:
                x_eta = self._eq_half_step(x_eta, half_dt, add_noise=not is_final_step)
            else:
                x_eta = self._analytic_half_step(x_eta, half_dt, add_noise=not is_final_step)

            t_score = (t_curr - half_dt) if score_time == "midpoint" else t_curr
            t_tensor = torch.full((n_samples,), t_score.item(), device=device)
            model_input = x_eta.clone()
            model_input[:, [1, 3, 5]] = model_input[:, [1, 3, 5]] / scale
            model_output = self.model(model_input, self._normalize_time(t_tensor))
            if use_eq:
                force = self._eq_score_force(model_output, x_eta)
            else:
                force = self._score_force(model_output, Tp_tensor, Ta_tensor, tau_tensor)
            x_eta = x_eta + force * dt

            if use_eq:
                x_eta = self._eq_half_step(x_eta, half_dt, add_noise=not is_final_step)
            else:
                x_eta = self._analytic_half_step(x_eta, half_dt, add_noise=not is_final_step)

        if tweedie:
            x_eta = self._active_tweedie_correction(x_eta, time_grid[-1].item(), device)

        final_rgb = x_eta[:, [0, 2, 4]]
        final_eta = x_eta[:, [1, 3, 5]]

        final_rgb = final_rgb.clip(-1, 1)
        final_rgb = (final_rgb + 1.) / 2.

        return final_rgb.detach(), final_eta.detach()

    @torch.no_grad()
    def sampling(self, n_samples, device="cuda", probability_flow=False, pf_steps=None, pf_schedule="linear", pf_solver="heun", pf_rtol=1e-4, pf_atol=1e-5, tweedie=True, t_end=None):
        """Generate samples - EXACTLY like MNIST but for RGB

        `t_end` sets where the time grid stops (default 1e-3); pass 0.0 to integrate
        straight to t=0, which is cheap for the active model (Tp ~ 1e-3 leaves almost
        no residual x-noise). Tweedie is skipped automatically when t_end <= 0.
        """
        if t_end is not None and float(t_end) <= 0.0:
            tweedie = False
        # Create initial shape for samples (6 channels: RGB + 3 eta)
        image_shape = (n_samples, 6, self.image_size, self.image_size)
        
        # Compute base covariance matrix
        base_cov = self.compute_covariance(self.T * torch.ones(n_samples, device=device))
        
        # Generate initial correlated noise
        x_eta = self.generate_correlated_noise(base_cov, num_channels=3)
        
        # Convert parameters to tensors
        Tp_tensor = torch.tensor(self.Tp, device=device)
        Ta_tensor = torch.tensor(self.Ta, device=device)
        tau_tensor = torch.tensor(self.tau, device=device)
        force_coeff_rgb = 2 * Tp_tensor
        force_coeff_eta = 2 * Ta_tensor / (tau_tensor * tau_tensor)

        if probability_flow and pf_solver == "rk45":
            t0 = float(self.time_range)
            t1 = 0.0
            steps = self.timesteps if (pf_steps is None or pf_steps <= 0) else int(pf_steps)
            h0 = (t0 - t1) / float(max(1, steps))
            max_step_abs = abs(h0)

            scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)

            def drift_fn(z, t_scalar: float):
                t_tensor = torch.full((n_samples,), float(t_scalar), device=device)
                model_input = z.clone()
                model_input[:, [1, 3, 5]] = model_input[:, [1, 3, 5]] / scale
                model_output = self.model(model_input, self._normalize_time(t_tensor))
                return self._active_pf_drift(z, model_output, Tp_tensor, Ta_tensor, tau_tensor)

            x_eta = _rk45_integrate(
                drift_fn,
                x_eta,
                t0=t0,
                t1=t1,
                h0=h0,
                rtol=pf_rtol,
                atol=pf_atol,
                max_step_abs=max_step_abs,
            )
        elif probability_flow:
            # Heun-Kutta (HK) integration for active PF-ODE (with scaled eta input to the UNet)
            time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, x_eta.device, t_end=t_end)
            for idx in tqdm(range(len(time_grid) - 1), desc="Sampling (PF-ODE, HK)"):
                t_curr = time_grid[idx]
                t_next = time_grid[idx + 1]
                dt = (t_curr - t_next).clamp(min=1e-6)

                # First stage (Euler predictor at t_curr)
                t_tensor = torch.full((n_samples,), t_curr.item(), device=device)
                scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)
                model_input = x_eta.clone()
                model_input[:, [1,3,5]] = model_input[:, [1,3,5]] / scale
                model_output = self.model(model_input, self._normalize_time(t_tensor))
                drift_curr = self._active_pf_drift(x_eta, model_output, Tp_tensor, Ta_tensor, tau_tensor)

                x_pred = x_eta + drift_curr * dt  # Euler prediction to t_next

                # Second stage (corrector using drift at t_next)
                t_next_tensor = torch.full((n_samples,), t_next.item(), device=device)
                model_input_next = x_pred.clone()
                model_input_next[:, [1,3,5]] = model_input_next[:, [1,3,5]] / scale
                model_output_next = self.model(model_input_next, self._normalize_time(t_next_tensor))
                drift_next = self._active_pf_drift(x_pred, model_output_next, Tp_tensor, Ta_tensor, tau_tensor)

                # Heun update (average of drifts)
                x_eta = x_eta + 0.5 * (drift_curr + drift_next) * dt

            if tweedie:
                x_eta = self._active_tweedie_correction(x_eta, time_grid[-1].item(), device)
        else:
            # SDE (Euler-Maruyama) sampler with a configurable step count/schedule
            # (reuses the same pf_steps/pf_schedule knobs as the PF-ODE branch; at the
            # default pf_steps=None this is effectively the old fixed-self.dt loop,
            # off by a single negligible step at the t=T boundary).
            time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, x_eta.device, t_end=t_end)
            n_intervals = len(time_grid) - 1
            for idx in tqdm(range(n_intervals), desc="Sampling"):
                t_curr = time_grid[idx]
                t_next = time_grid[idx + 1]
                dt = (t_curr - t_next).clamp(min=1e-6)
                dt_tensor = dt.to(device)
                noise_scale_rgb = torch.sqrt(2 * Tp_tensor * dt_tensor)
                noise_scale_eta = (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor)

                t_tensor = torch.full((n_samples,), t_curr.item(), device=device)

                scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)
                model_input = x_eta.clone()
                model_input[:, [1,3,5]] = model_input[:, [1,3,5]] / scale
                model_output = self.model(model_input, self._normalize_time(t_tensor))

                is_final_step = tweedie and (idx == n_intervals - 1)

                for channel in range(3):
                    rgb_idx = channel * 2
                    eta_idx = channel * 2 + 1

                    rgb = x_eta[:, rgb_idx:rgb_idx+1]
                    eta = x_eta[:, eta_idx:eta_idx+1]
                    F_rgb = model_output[:, rgb_idx:rgb_idx+1]
                    F_eta = model_output[:, eta_idx:eta_idx+1]

                    rgb = rgb + dt_tensor * (self.k * rgb - eta) + force_coeff_rgb * F_rgb * dt_tensor
                    if not is_final_step:
                        rgb = rgb + noise_scale_rgb * torch.randn_like(rgb)

                    eta = eta + (dt_tensor * eta / tau_tensor) + \
                          force_coeff_eta * F_eta * dt_tensor
                    if not is_final_step:
                        eta = eta + noise_scale_eta * torch.randn_like(eta)

                    x_eta[:, rgb_idx:rgb_idx+1] = rgb
                    x_eta[:, eta_idx:eta_idx+1] = eta

            if tweedie:
                x_eta = self._active_tweedie_correction(x_eta, time_grid[-1].item(), device)

        # Extract final RGB channels
        final_rgb = x_eta[:, [0,2,4]]  # R, G, B channels
        final_eta = x_eta[:, [1,3,5]]  # η_r, η_g, η_b channels
        
        # Scale to [0, 1] range - EXACTLY like MNIST
        final_rgb = final_rgb.clip(-1, 1)
        final_rgb = (final_rgb + 1.) / 2.
        
        return final_rgb.detach(), final_eta.detach() 

    @torch.no_grad()
    def sampling_from_intermediate(self, x_eta, t_start, device="cuda", probability_flow=False, pf_steps=None, pf_schedule="linear", pf_solver="heun", pf_rtol=1e-4, pf_atol=1e-5):
        """
        Continue reverse-time sampling starting from an intermediate state x_eta.
        Args:
            x_eta: tensor [batch, 6, H, W] containing interleaved RGB and eta channels
            t_start: integer timestep index to resume from
            device: device to run sampling on
        Returns:
            Tuple of (final_rgb, final_eta)
        """
        batch_size = x_eta.shape[0]
        Tp_tensor = torch.tensor(self.Tp, device=device)
        Ta_tensor = torch.tensor(self.Ta, device=device)
        tau_tensor = torch.tensor(self.tau, device=device)
        dt_tensor = torch.tensor(self.dt, device=device)
        adjusted_t_start = min(t_start, self.timesteps - 2)

        noise_scale_rgb = torch.sqrt(2 * Tp_tensor * dt_tensor)
        noise_scale_eta = (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor)
        force_coeff_rgb = 2 * Tp_tensor
        force_coeff_eta = 2 * Ta_tensor / (tau_tensor * tau_tensor)

        if probability_flow and pf_solver == "rk45":
            t0 = float(adjusted_t_start * self.dt)
            t1 = 0.0
            steps = self.timesteps if (pf_steps is None or pf_steps <= 0) else int(pf_steps)
            h0 = (t0 - t1) / float(max(1, steps))
            max_step_abs = abs(h0)

            scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)

            def drift_fn(z, t_scalar: float):
                t_tensor = torch.full((batch_size,), float(t_scalar), device=device)
                model_input = z.clone()
                model_input[:, [1, 3, 5]] = model_input[:, [1, 3, 5]] / scale
                model_output = self.model(model_input, self._normalize_time(t_tensor))
                return self._active_pf_drift(z, model_output, Tp_tensor, Ta_tensor, tau_tensor)

            x_eta = _rk45_integrate(
                drift_fn,
                x_eta,
                t0=t0,
                t1=t1,
                h0=h0,
                rtol=pf_rtol,
                atol=pf_atol,
                max_step_abs=max_step_abs,
            )
        elif probability_flow:
            start_time = adjusted_t_start * self.dt
            time_grid = self._build_pf_time_grid(pf_steps, pf_schedule, x_eta.device, start_time=start_time)
            for idx in range(len(time_grid) - 1):
                t_curr = time_grid[idx]
                t_next = time_grid[idx + 1]
                dt = (t_curr - t_next).clamp(min=1e-6)

                t_tensor = torch.full((batch_size,), t_curr.item(), device=device)
                scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)
                model_input = x_eta.clone()
                model_input[:, [1,3,5]] = model_input[:, [1,3,5]] / scale
                model_output = self.model(model_input, self._normalize_time(t_tensor))
                drift_curr = self._active_pf_drift(x_eta, model_output, Tp_tensor, Ta_tensor, tau_tensor)
                x_euler = x_eta + drift_curr * dt

                t_next_tensor = torch.full((batch_size,), t_next.item(), device=device)
                model_input_next = x_euler.clone()
                model_input_next[:, [1,3,5]] = model_input_next[:, [1,3,5]] / scale
                model_output_next = self.model(model_input_next, self._normalize_time(t_next_tensor))
                drift_next = self._active_pf_drift(x_euler, model_output_next, Tp_tensor, Ta_tensor, tau_tensor)

                x_eta = x_eta + 0.5 * dt * (drift_curr + drift_next)
        else:
            for t in range(adjusted_t_start, -1, -1):
                t_tensor = (t * self.dt) * torch.ones((batch_size,), device=device)
                # Keep eta scaling consistent with the main sampler (important for resumed sampling).
                scale = self.eta_scale_tensor.view(1, 1, 1, 1).to(x_eta.device)
                model_input = x_eta.clone()
                model_input[:, [1, 3, 5]] = model_input[:, [1, 3, 5]] / scale
                model_output = self.model(model_input, self._normalize_time(t_tensor))

                # Determine if this is the final step (t=0)
                is_final_step = (t == 0)

                for channel in range(3):
                    rgb_idx = channel * 2
                    eta_idx = channel * 2 + 1

                    rgb = x_eta[:, rgb_idx:rgb_idx+1]
                    eta = x_eta[:, eta_idx:eta_idx+1]
                    F_rgb = model_output[:, rgb_idx:rgb_idx+1]
                    F_eta = model_output[:, eta_idx:eta_idx+1]

                    rgb = rgb + dt_tensor * (self.k * rgb - eta) + force_coeff_rgb * F_rgb * dt_tensor
                    if not is_final_step:
                        rgb = rgb + noise_scale_rgb * torch.randn_like(rgb, device=device)
                    
                    eta = eta + (dt_tensor * eta / tau_tensor) + \
                          force_coeff_eta * F_eta * dt_tensor
                    if not is_final_step:
                        eta = eta + noise_scale_eta * torch.randn_like(eta, device=device)

                    x_eta[:, rgb_idx:rgb_idx+1] = rgb
                    x_eta[:, eta_idx:eta_idx+1] = eta

        final_rgb = x_eta[:, [0, 2, 4]]
        final_eta = x_eta[:, [1, 3, 5]]
        final_rgb = final_rgb.clip(-1, 1)
        final_rgb = (final_rgb + 1.) / 2.

        return final_rgb.detach(), final_eta.detach()