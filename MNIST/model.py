import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm
import numpy as np

def manual_cholesky(covariance_matrix):
    """
    Manually compute Cholesky decomposition for batch of 2x2 matrices
    Input: covariance_matrix with shape [batch_size, 2, 2]
    Output: L with shape [batch_size, 2, 2] where L is lower triangular
    With protection against negative values under square roots
    """
    batch_size = covariance_matrix.shape[0]
    device = covariance_matrix.device
    
    # Initialize L with zeros
    L = torch.zeros((batch_size, 2, 2), device=device)
    
    # Extract components from covariance matrix
    M11 = covariance_matrix[:, 0, 0]  # shape: [batch_size]
    M12 = covariance_matrix[:, 0, 1]  # shape: [batch_size]
    M22 = covariance_matrix[:, 1, 1]  # shape: [batch_size]
    
    # Compute L components with protection against negative values
    M11_safe = torch.clamp(M11, min=0.0)     # ensure M11 >= 0
    l11 = torch.sqrt(M11_safe)               # l11 = √max(M11, 0)
    l12 = torch.zeros_like(M11)              # l12 = 0
    
    # Compute l21 (protect against division by zero)
    l21 = torch.where(l11 > 0, M12 / l11, torch.zeros_like(M12))
    
    # Compute l22 with protection against negative values
    l22_term = torch.clamp(M22 - l21 * l21, min=0.0)  # ensure M22 - l21² >= 0
    l22 = torch.sqrt(l22_term)               # l22 = √max(M22 - l21², 0)
    
    # Assign values to L
    L[:, 0, 0] = l11
    L[:, 0, 1] = l12
    L[:, 1, 0] = l21
    L[:, 1, 1] = l22
    
    return L

class MNISTDiffusion(nn.Module):
    def __init__(self, image_size, in_channels, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8], T=1.0, k=1.0, total_time=2.0):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.dt = total_time/timesteps
        self.T = T
        self.k = k
        self.total_time = total_time

        self.model = Unet(timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults)

    def marginal_prob(self, x_0, t):
        """
        Returns mean and standard deviation of the marginal distribution p(x_t|x_0)
        for the passive diffusion SDE:
        dx = -k x dt + sqrt(2T) dW
        """
        mean = x_0 * torch.exp(-self.k * t)[:, None, None, None]
        std = torch.sqrt((self.T / self.k) * (1 - torch.exp(-2 * self.k * t)))[:, None, None, None]
        return mean, std

    def forward(self, x, noise):
        # Generate random time between 0 and total_time
        t = 1e-3 + (self.total_time - 1e-3) * torch.rand(x.shape[0], device=x.device)
        
        # Get noisy image and score
        mean, std = self.marginal_prob(x, t)
        x_t = mean + std * noise
        score = self.model(x_t, t)
        
        # Return for loss computation
        return x_t, score, mean, std
    
    @torch.no_grad()
    def sampling(self, n_samples, device="cuda"):
        """
        Generate samples using Euler-Maruyama solver for the reverse-time SDE
        """
        # Initialize from stationary distribution N(0, T/k)
        x_t = math.sqrt(self.T / self.k) * torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size), device=device
        )

        # Reverse-time SDE: dx = [k x + 2T score] dt + sqrt(2T) dW
        noise_scale = math.sqrt(2 * self.T * self.dt)
        for i in tqdm(range(self.timesteps-1, -1, -1), desc="Sampling"):
            t = torch.tensor([i*self.dt for _ in range(n_samples)]).to(device)
            score = self.model(x_t, t)
            drift = -self.k * x_t - 2 * self.T * score
            x_t = x_t - drift * self.dt + noise_scale * torch.randn_like(x_t)
        
        # Scale to [0, 1] range
        x_t = x_t.clip(-1, 1)
        x_t = (x_t + 1.) / 2.
        return x_t
        
    @torch.no_grad()
    def sampling_from_intermediate(self, x_t, t_step, device="cuda"):
        """
        Generate samples starting from an intermediate state x_t at timestep t_step.
        
        Args:
            x_t: Tensor of shape [batch_size, channels, height, width]
            t_step: The timestep corresponding to the current state
            device: Device to run the sampling on
            
        Returns:
            x: The final reconstructed image
        """
        n_samples = x_t.shape[0]
        
        # Reverse-time SDE: dx = [k x + 2T score] dt + sqrt(2T) dW
        noise_scale = math.sqrt(2 * self.T * self.dt)
        for i in tqdm(range(t_step, -1, -1), desc="Sampling from intermediate"):
            t = torch.tensor([i*self.dt for _ in range(n_samples)]).to(device)
            score = self.model(x_t, t)
            drift = -self.k * x_t - 2 * self.T * score
            x_t = x_t - drift * self.dt + noise_scale * torch.randn_like(x_t)
        
        # Scale to [0, 1] range
        x_t = x_t.clip(-1, 1)
        x_t = (x_t + 1.) / 2.
        return x_t
    


class MNIST_Active_Diffusion(nn.Module):
    def __init__(self, image_size, time_embedding_dim=256, timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8], Tp=1e-3, Ta=1.0, k=1.0, tau=0.1, total_time=2.0):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = 2  # Still need 2 input channels (x and eta)
        self.image_size = image_size
        self.dt = total_time/timesteps
        
        # Physical parameters
        self.Tp = Tp
        self.Ta = Ta
        self.k = k
        self.tau = tau
        self.total_time = total_time
        
        # Model now outputs 2 channels for F_x and F_eta
        self.model = Unet(timesteps, time_embedding_dim, self.in_channels, 2, base_dim, dim_mults)

    def generate_eta0(self, batch_size, device=None):
        """
        Generate eta_0 ~ N(0, Ta/tau) — stationary marginal variance of eta.
        """
        std = math.sqrt(self.Ta / self.tau)
        return std * torch.randn(batch_size, 1, self.image_size, self.image_size, device=device)

    def compute_covariance(self, t):
        """
        Compute the covariance matrix for the given time points.
        """
        batch_size = t.shape[0]
        Tp = torch.full((batch_size,), self.Tp, device=t.device)
        Ta = torch.full((batch_size,), self.Ta, device=t.device)
        k = torch.full((batch_size,), self.k, device=t.device)
        tau = torch.full((batch_size,), self.tau, device=t.device)
        
        # Compute intermediate terms
        a = torch.exp(-k * t)
        b = torch.exp(-t / tau)
        Tx = Tp
        Ty = Ta / (tau * tau)
        w = 1 / tau
        
        # Compute matrix elements
        M11 = (1/k)*Tx*(1-a*a) + (1/k)*Ty*(
            1/(w*(k+w)) + 
            4*a*b*k/((k+w)*(k-w)**2) - 
            (k*b*b + w*a*a)/(w*(k-w)**2) + 1e-8
        )
        M12 = (Ty/(w*(k*k - w*w))) * (k*(1-b*b) - w*(1 + b*b - 2*a*b))
        M22 = (Ty/w)*(1-b*b) + 1e-8
        
        # Stack elements to form covariance matrix
        cov_matrix = torch.stack([
            torch.stack([M11, M12], dim=1),  # First row: [M11, M12]
            torch.stack([M12, M22], dim=1)   # Second row: [M12, M22]
        ], dim=1)
        
        return cov_matrix

    
    def generate_correlated_noise(self, covariance_matrix, image_shape):
        
        #Generate correlated Gaussian noise for MNIST-shaped data.
        
        batch_size = covariance_matrix.shape[0]
        height, width = image_shape[-2], image_shape[-1]
        covariance_matrix = covariance_matrix.view(batch_size, 2, 2)
        
        # Perform Cholesky decomposition
        L = torch.linalg.cholesky(covariance_matrix)
        #L = manual_cholesky(covariance_matrix)
        # Generate uncorrelated standard normal noise
        uncorrelated_noise = torch.randn(batch_size, 2, height, width).to(covariance_matrix.device)
        
        # Reshape for batch matrix multiplication
        uncorrelated_noise = uncorrelated_noise.view(batch_size, 2, -1)
        
        # Apply the Cholesky factor to obtain correlated noise
        correlated_noise = torch.bmm(L, uncorrelated_noise)
        
        # Reshape back to image format
        correlated_noise = correlated_noise.view(batch_size, 2, height, width)
        
        return correlated_noise
    
    """
    def generate_correlated_noise(self, covariance_matrix, spatial_shape):
        
        #Generate correlated Gaussian noise using MultivariateNormal distribution (vectorized)
        #Args:
        #    covariance_matrix: tensor of shape [batch_size, 2, 2]
        #    spatial_shape: tuple of (channels, height, width) for the spatial dimensions
        #Returns:
        #    noise: tensor of shape [batch_size, 2, height, width]
        
        batch_size = covariance_matrix.shape[0]
        height, width = spatial_shape[-2:]
        device = covariance_matrix.device
        
        # Create zero mean vectors for all samples at once
        loc = torch.zeros(batch_size, 2, device=device)
        
        # Create distribution for all batch items
        dist = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix
        )
        
        # Generate all samples at once: [batch_size, 2] x (height * width)
        samples = dist.sample((height * width,))  # shape: [height*width, batch_size, 2]
        
        # Reshape to desired output format
        noise = samples.permute(1, 2, 0).reshape(batch_size, 2, height, width)
        
        return noise
    """

    def compute_mean(self, x_0, eta_0, t):
        """
        Compute mean values at time t
        """
        k = self.k
        tau = self.tau
        
        # Reshape t for broadcasting
        t_view = t.view(-1, 1, 1, 1)
        
        # Compute coefficients
        a = torch.exp(-k * t_view)
        b = (torch.exp(-t_view / tau) - torch.exp(-k * t_view)) / (k - (1 / tau))
        c = torch.exp(-t_view / tau)
        
        # Compute means
        mean_x = a * x_0 + b * eta_0
        mean_eta = c * eta_0
        
        return mean_x, mean_eta

    def marginal_prob(self, x_0, eta_0, t):
        """
        Returns mean and covariance matrix of the marginal distribution p(x_t, eta_t|x_0, eta_0)
        """
        # Compute means
        mean_x, mean_eta = self.compute_mean(x_0, eta_0, t)
        
        # Compute covariance matrix
        cov = self.compute_covariance(t)
        
        return (mean_x, mean_eta), cov

    def forward(self, x, t=None, noise=None):
        """
        Forward pass of the model.
        """
        # Split input channels
        batch_size = x.shape[0]
        x_0 = x[:, 0:1]  # First channel: image
        eta_0 = x[:, 1:2]  # Second channel: eta_0
        
        # Generate random time if not provided
        if t is None:
            t = 1e-3 + (self.total_time - 1e-3) * torch.rand(batch_size, device=x.device)
        
        # Get means and covariance
        (mean_x, mean_eta), cov = self.marginal_prob(x_0, eta_0, t)
        
        # Generate correlated noise if not provided
        if noise is None:
            noise = self.generate_correlated_noise(cov, x_0.shape[-3:])
        
        # Extract noise components
        noise_x = noise[:, 0:1]
        noise_eta = noise[:, 1:2]
        
        # Compute noisy samples
        x_t = mean_x + noise_x
        eta_t = mean_eta + noise_eta
        
        # Get F_x and F_eta from model (using concatenated state)
        model_input = torch.cat([x_t, eta_t], dim=1)
        model_output = self.model(model_input, t)  # Now outputs 2 channels
        F_x = model_output[:, 0:1]  # First channel for F_x
        F_eta = model_output[:, 1:2]  # Second channel for F_eta
        
        # Return noise along with other outputs for consistent loss calculation
        return (x_t, eta_t), (F_x, F_eta), (mean_x, mean_eta), cov, noise

    def diffusion_loss_active(self, model_output, x0, eta0, t, noise=None):
        """
        Compute the diffusion loss for active particles.
        """
        device = x0.device
        t = t.view(-1, 1, 1, 1)
        
        # Split model output into F_x and F_eta
        F_x, F_eta = model_output
        
        # Compute mean
        mean_x, mean_eta = self.compute_mean(x0, eta0, t)
        
        # Compute covariance matrix
        M = self.compute_covariance(t.squeeze())
        
        # Use the same noise from forward pass if provided
        if noise is None:
            # If noise wasn't provided, generate it (same as in forward)
            noise = self.generate_correlated_noise(M, x0.shape[-3:])
        
        # Compute noisy samples using the same noise
        xt = mean_x + noise[:, 0:1]
        etat = mean_eta + noise[:, 1:2]
        
        # Extract covariance matrix components
        M11 = M[:, 0, 0].view(-1, 1, 1, 1)
        M12 = M[:, 0, 1].view(-1, 1, 1, 1)
        M22 = M[:, 1, 1].view(-1, 1, 1, 1)
        det = M11 * M22 - M12 * M12
        
        # Compute mean coefficients
        k = self.k
        tau = self.tau
        a = torch.exp(-k * t)
        b = (torch.exp(-t / tau) - torch.exp(-k * t)) / (k - (1 / tau))
        c = torch.exp(-t / tau)
        
        # Compute loss terms
        Feta = torch.sqrt(1 / det) * (-M11 * (etat - c * eta0) + M12 * (xt - a * x0 - b * eta0))
        Fx = torch.sqrt(1 / det) * (-M22 * (xt - a * x0 - b * eta0) + M12 * (etat - c * eta0))
        
        scr_eta = torch.sqrt(det) * F_eta
        scr_x = torch.sqrt(det) * F_x
        
        
        # Compute loss terms
        # Original loss terms
        #Feta = -M11 * (etat - c * eta0) + M12 * (xt - a * x0 - b * eta0)
        #Fx =  -M22 * (xt - a * x0 - b * eta0) + M12 * (etat - c * eta0)
        
        # Modified loss terms - excluding eta0 terms
        #Feta = -M11 * etat + M12 * (xt - a * x0)
        #Fx =  -M22 * (xt - a * x0) + M12 * etat
        
        #scr_eta = det * F_eta
        #scr_x = det * F_x
        
        # Compute combined loss
        loss_eta = torch.mean((scr_eta - Feta) ** 2)
        loss_x = torch.mean((scr_x - Fx) ** 2)
        
        # Return total loss
        return self.Ta * loss_eta + self.Tp * loss_x

    @torch.no_grad()
    def sampling(self, n_samples, device="cuda"):
        """
        Generate samples using the active sampling procedure.
        """
        # Create initial shape for samples
        image_shape = (n_samples, self.in_channels, self.image_size, self.image_size)
        
        # Compute base covariance matrix
        base_cov = self.compute_covariance(self.total_time*torch.ones(n_samples, device=device))
        
        # Generate initial correlated noise
        x_eta = self.generate_correlated_noise(base_cov, image_shape[1:])
        
        # Convert parameters to tensors
        Tp_tensor = torch.tensor(self.Tp, device=device)
        Ta_tensor = torch.tensor(self.Ta, device=device)
        tau_tensor = torch.tensor(self.tau, device=device)
        dt_tensor = torch.tensor(self.dt, device=device)
        k_tensor = torch.tensor(self.k, device=device)
        
        # Generate eta_0 for correction terms (now using fixed value of Ta/tau)
        eta_0 = self.generate_eta0(n_samples, device=device)
        
        # Main sampling loop
        for t in tqdm(range(self.timesteps - 2, 0, -1), desc="Sampling"):
            # Create time tensor
            t_tensor = (t * self.dt) * torch.ones((n_samples,), device=device)
            
            # Extract x and eta components
            x = x_eta[:, 0:1]  # Shape: [batch_size, 1, height, width]
            eta = x_eta[:, 1:2]  # Shape: [batch_size, 1, height, width]
            
            # Compute coefficients for correction terms
            t_view = t_tensor.view(-1, 1, 1, 1)
            a = torch.exp(-k_tensor * t_view)
            b = (torch.exp(-t_view / tau_tensor) - torch.exp(-k_tensor * t_view)) / (k_tensor - (1 / tau_tensor))
            c = torch.exp(-t_view / tau_tensor)
            
            # Compute covariance matrix components for correction terms
            M = self.compute_covariance(t_tensor)
            M11 = M[:, 0, 0].view(-1, 1, 1, 1)
            M12 = M[:, 0, 1].view(-1, 1, 1, 1)
            M22 = M[:, 1, 1].view(-1, 1, 1, 1)
            det = M11 * M22 - M12 * M12
            
            # Get model predictions F_x and F_eta
            with torch.no_grad():
                model_input = torch.cat([x, eta], dim=1)
                model_output = self.model(model_input, t_tensor)
                F_x = model_output[:, 0:1]
                F_eta = model_output[:, 1:2]
            
            # Update x using the active sampling procedure
            x = x + dt_tensor * (k_tensor * x - eta) + (2 * Tp_tensor) * model_output[:,0:1] * dt_tensor + \
                  torch.sqrt(2 * Tp_tensor * dt_tensor) * torch.randn_like(x, device=device)
            
            # Update eta using the active sampling procedure
            eta = eta + (dt_tensor * eta / tau_tensor) + \
                  (2 * Ta_tensor / (tau_tensor * tau_tensor)) * model_output[:, 1:2] * dt_tensor + \
                  (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor) * torch.randn_like(eta, device=device)
            
            
            # Calculate correction terms that were excluded from the loss
            #correction_x = (1 / det) * (-M22 * (-b * eta_0) + M12 * (-c * eta_0))
            #correction_eta = (1 / det) * (-M11 * (-c * eta_0) + M12 * (-b * eta_0))
                        
            # Modified update equations with correction terms
            # Original x update
            #x = x + dt_tensor * (x - eta) + (2 * Tp_tensor) * model_output[:,0:1] * dt_tensor + \
            #       torch.sqrt(2 * Tp_tensor * dt_tensor) * torch.randn_like(x, device=device)
            
            # Original eta update
            #eta = eta + (dt_tensor * eta / tau_tensor) + \
            #      (2 * Ta_tensor / (tau_tensor * tau_tensor)) * model_output[:, 1:2] * dt_tensor + \
            #      (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor) * torch.randn_like(eta, device=device)
            
            # Modified x update with correction
            #x = x + dt_tensor * (x - eta) + \
            #    (2 * Tp_tensor) * (F_x + correction_x) * dt_tensor + \
            #    torch.sqrt(2 * Tp_tensor * dt_tensor) * torch.randn_like(x, device=device)
            
            # Modified eta update with correction
            #eta = eta + (dt_tensor * eta / tau_tensor) + \
            #      (2 * Ta_tensor / (tau_tensor * tau_tensor)) * (F_eta + correction_eta) * dt_tensor + \
            #      (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor) * torch.randn_like(eta, device=device)
            
            # Recombine x and eta
            x_eta = torch.cat((x, eta), dim=1)
        
        # Scale the final image to [0, 1] range
        x = x_eta[:, 0:1]
        x = x.clip(-1, 1)
        x = (x + 1.) / 2.
        
        return x.detach(), x_eta[:, 1:2].detach()
        
    @torch.no_grad()
    def sampling_from_intermediate(self, x_eta_t, t_step, device="cuda"):
        """
        Generate samples starting from an intermediate state x_eta_t at timestep t_step.
        
        Args:
            x_eta_t: Tensor of shape [batch_size, 2, height, width] containing x_t and eta_t
            t_step: The timestep corresponding to the current state
            device: Device to run the sampling on
        
        Returns:
            x: The final reconstructed image
            eta: The final eta component
        """
        n_samples = x_eta_t.shape[0]
        
        # Convert parameters to tensors
        Tp_tensor = torch.tensor(self.Tp, device=device)
        Ta_tensor = torch.tensor(self.Ta, device=device)
        tau_tensor = torch.tensor(self.tau, device=device)
        dt_tensor = torch.tensor(self.dt, device=device)
        k_tensor = torch.tensor(self.k, device=device)
        
        # Generate eta_0 for correction terms
        eta_0 = self.generate_eta0(n_samples, device=device)
        
        # Main sampling loop - start from t_step and go backwards
        for t in tqdm(range(t_step, 0, -1), desc="Sampling from intermediate"):
            # Create time tensor
            t_tensor = (t * self.dt) * torch.ones((n_samples,), device=device)
            
            # Extract x and eta components
            x = x_eta_t[:, 0:1]  # Shape: [batch_size, 1, height, width]
            eta = x_eta_t[:, 1:2]  # Shape: [batch_size, 1, height, width]
            
            # Compute coefficients for correction terms
            t_view = t_tensor.view(-1, 1, 1, 1)
            a = torch.exp(-k_tensor * t_view)
            b = (torch.exp(-t_view / tau_tensor) - torch.exp(-k_tensor * t_view)) / (k_tensor - (1 / tau_tensor))
            c = torch.exp(-t_view / tau_tensor)
            
            # Compute covariance matrix components for correction terms
            M = self.compute_covariance(t_tensor)
            M11 = M[:, 0, 0].view(-1, 1, 1, 1)
            M12 = M[:, 0, 1].view(-1, 1, 1, 1)
            M22 = M[:, 1, 1].view(-1, 1, 1, 1)
            det = M11 * M22 - M12 * M12
            
            # Get model predictions F_x and F_eta
            model_input = torch.cat([x, eta], dim=1)
            model_output = self.model(model_input, t_tensor)
            #F_x = model_output[:, 0:1]
            #F_eta = model_output[:, 1:2]
            # Update x using the active sampling procedure
            x_drift = k_tensor * x - eta
            x_score = model_output[:, 0:1]
            x_noise = torch.randn_like(x, device=device)
            
            x = x + dt_tensor * x_drift + \
                  2 * Tp_tensor * dt_tensor * x_score + \
                  torch.sqrt(2 * Tp_tensor * dt_tensor) * x_noise
            
            # Update eta using the active sampling procedure
            eta_drift = eta / tau_tensor
            eta_score = model_output[:, 1:2]
            eta_noise = torch.randn_like(eta, device=device)
            
            eta = eta + dt_tensor * eta_drift + \
                  (2 * Ta_tensor / (tau_tensor * tau_tensor)) * dt_tensor * eta_score + \
                  (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor) * eta_noise
            
            # Recombine x and eta
            x_eta = torch.cat((x, eta), dim=1)
        
            # Calculate correction terms that were excluded from the loss
            #correction_x = (1 / det) * (-M22 * (-b * eta_0) + M12 * (-c * eta_0))
            #correction_eta = (1 / det) * (-M11 * (-c * eta_0) + M12 * (-b * eta_0))
            
            # Modified x update with correction
            #x = x + dt_tensor * (x - eta) + \
            #    (2 * Tp_tensor) * (F_x + correction_x) * dt_tensor + \
            #    torch.sqrt(2 * Tp_tensor * dt_tensor) * torch.randn_like(x, device=device)
            
            # Modified eta update with correction
            #eta = eta + (dt_tensor * eta / tau_tensor) + \
            #      (2 * Ta_tensor / (tau_tensor * tau_tensor)) * (F_eta + correction_eta) * dt_tensor + \
            #      (1 / tau_tensor) * torch.sqrt(2 * Ta_tensor * dt_tensor) * torch.randn_like(eta, device=device)
            
            # Recombine x and eta
            x_eta_t = torch.cat((x, eta), dim=1)
        
        # Scale the final image to [0, 1] range
        x = x_eta_t[:, 0:1]
        x = x.clip(-1, 1)
        x = (x + 1.) / 2.
        
        return x.detach(), x_eta_t[:, 1:2].detach()


import math
import torch
import torch.nn as nn
from tqdm import tqdm
from unet import Unet


class MNIST_Linear_Diffusion(nn.Module):
    """
    Generic 2D linear diffusion model with arbitrary 2x2 drift M and noise D.

    Forward SDE:
        dz = M z dt + D dW,    where z = (x, eta)

    For stability, eigenvalues of M must have negative real parts. The
    stationary distribution is N(0, C_inf) where C_inf solves the Lyapunov
    equation
        M C_inf + C_inf M^T = - D D^T.

    The model predicts the uncorrelated noise eps such that
        z_t = mean(t) + L(t) eps,   L(t) = chol(Q(t))
    The score is recovered as
        grad log p = - L(t)^{-T} eps_pred.

    Args:
        image_size: spatial size of the data (assumed square).
        M_init: length-4 list/tensor of initial drift entries [m11, m12, m21, m22].
        D: length-4 list/tensor of noise entries [d11, d12, d21, d22].
        learn_M: if True, M entries become learnable parameters.
        M_constraint: how to constrain M during learning. One of:
            'none' — no norm constraint (only L2 regularizer and stability
                penalty). This is the default but may allow entries to grow.
            'fixed_norm' — reparameterize M = r * M_hat / ||M_hat||_F so
                that ||M||_F = r exactly at all times. Only the direction
                on the matrix sphere is learned. Set r via M_norm_radius
                (defaults to ||M_init||_F).
            'clamp_norm' — after each optimizer step, if ||M||_F > max_norm,
                rescale M so ||M||_F = max_norm. Set via M_max_norm
                (defaults to 2 * ||M_init||_F).
        M_norm_radius: radius r for 'fixed_norm' mode. If None, uses
            ||M_init||_F.
        M_max_norm: ceiling for 'clamp_norm' mode. If None, uses
            2 * ||M_init||_F.
        stability_weight: coefficient lambda for the stability penalty term.
        M_l2_weight: coefficient for L2 (Frobenius norm) regularization on M.
        eta0_mode: how the initial eta channel is generated. One of:
            'stationary_marginal' — draw eta_0 ~ N(0, (C_inf)_22) per pixel.
            'conditional' — draw eta_0 | x_0 from the conditional of N(0, C_inf).
            'constant' — fill eta_0 with eta0_value.
            'zero' — eta_0 = 0.
        eta0_value: only used when eta0_mode == 'constant'.
    """

    def __init__(self, image_size, M_init, D,
                 learn_M=False, M_constraint='none',
                 M_norm_radius=None, M_max_norm=None,
                 stability_weight=10.0, M_l2_weight=0.01,
                 eta0_mode='stationary_marginal', eta0_value=0.0,
                 time_embedding_dim=256, timesteps=1000, base_dim=32,
                 dim_mults=[2, 4], total_time=2.0):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = 2
        self.image_size = image_size
        self.dt = total_time / timesteps
        self.total_time = total_time
        self.eta0_mode = eta0_mode
        self.eta0_value = float(eta0_value)
        self.learn_M = learn_M
        self.M_constraint = M_constraint
        self.stability_weight = stability_weight
        self.M_l2_weight = M_l2_weight

        # Coerce M and D into 2x2 float32 tensors
        if not isinstance(M_init, torch.Tensor):
            M_init = torch.tensor(M_init, dtype=torch.float32)
        if not isinstance(D, torch.Tensor):
            D = torch.tensor(D, dtype=torch.float32)
        M_init = M_init.reshape(2, 2).to(torch.float32)
        D = D.reshape(2, 2).to(torch.float32)

        init_norm = torch.linalg.norm(M_init, 'fro').item()

        # Stability check
        eigs = torch.linalg.eigvals(M_init).real
        if (eigs >= 0).any():
            print(f"[MNIST_Linear_Diffusion] WARNING: initial M may be "
                  f"unstable; Re(eigenvalues) = {eigs.tolist()}")

        # M: either learnable parameter or fixed buffer
        if learn_M:
            if M_constraint == 'fixed_norm':
                # Learn direction only; norm is fixed to radius r
                self.M_norm_radius = M_norm_radius if M_norm_radius is not None else init_norm
                self._M_raw = nn.Parameter(M_init.clone())
                print(f"[MNIST_Linear_Diffusion] fixed_norm mode: "
                      f"radius = {self.M_norm_radius:.4f}")
            elif M_constraint == 'clamp_norm':
                self.M_max_norm = M_max_norm if M_max_norm is not None else 2.0 * init_norm
                self.M = nn.Parameter(M_init.clone())
                print(f"[MNIST_Linear_Diffusion] clamp_norm mode: "
                      f"max_norm = {self.M_max_norm:.4f}")
            else:
                self.M = nn.Parameter(M_init.clone())
        else:
            self.register_buffer('M', M_init)

        self.register_buffer('D', D)
        self.register_buffer('Sigma', D @ D.T)

        self.model = Unet(timesteps, time_embedding_dim,
                          self.in_channels, 2, base_dim, dim_mults)

    # -------------------- M access with constraints ----------------------------

    def get_M(self):
        """
        Return the effective drift matrix, applying any constraint.
        For 'fixed_norm': M = radius * _M_raw / ||_M_raw||_F.
        For 'clamp_norm' and 'none': returns self.M directly.
        """
        if self.learn_M and self.M_constraint == 'fixed_norm':
            raw = self._M_raw
            norm = torch.linalg.norm(raw, 'fro').clamp(min=1e-8)
            return self.M_norm_radius * raw / norm
        return self.M

    def clamp_M_norm(self):
        """
        Call after optimizer.step() when using 'clamp_norm' mode.
        Projects M back to the ball ||M||_F <= max_norm in-place.
        """
        if self.learn_M and self.M_constraint == 'clamp_norm':
            with torch.no_grad():
                norm = torch.linalg.norm(self.M, 'fro')
                if norm > self.M_max_norm:
                    self.M.mul_(self.M_max_norm / norm)

    # -------------------- stationary covariance (dynamic) --------------------

    def compute_C_inf(self):
        """
        Solve the Lyapunov equation M C + C M^T = -Sigma for C_inf.
        Recomputed every call so gradients flow through M when learn_M=True.
        """
        M = self.get_M()
        I2 = torch.eye(2, device=M.device, dtype=M.dtype)
        A = torch.kron(I2, M) + torch.kron(M, I2)  # 4x4
        vecC = torch.linalg.solve(A, -self.Sigma.reshape(4))
        C = vecC.reshape(2, 2)
        C = 0.5 * (C + C.T)                                  # symmetrize
        C = C + 1e-6 * I2                                    # ridge
        return C

    def stability_penalty(self):
        """
        Soft penalty that pushes eigenvalues of M to have negative real parts.
        For 2x2: stability requires tr(M) < 0 and det(M) > 0.
        Returns a scalar loss >= 0 that is zero when M is well inside the
        stable region.
        """
        M = self.get_M()
        tr = M[0, 0] + M[1, 1]
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        # Penalize tr(M) >= -margin  (want tr << 0)
        # Penalize det(M) <= +margin (want det >> 0)
        margin = 0.1
        loss_tr = torch.relu(tr + margin) ** 2
        loss_det = torch.relu(margin - det) ** 2
        return loss_tr + loss_det

    # -------------------- eta_0 generation -----------------------------------

    def generate_eta0(self, batch_size, device=None, x_0=None):
        """
        Draw the initial eta_0 field according to eta0_mode.
        For 'conditional', x_0 must be passed (shape [B, 1, H, W]).
        """
        shape = (batch_size, 1, self.image_size, self.image_size)

        if self.eta0_mode == 'zero':
            return torch.zeros(shape, device=device)

        if self.eta0_mode == 'constant':
            return torch.full(shape, self.eta0_value, device=device)

        # Both remaining modes need C_inf
        C_inf = self.compute_C_inf()
        c11, c12, c22 = C_inf[0, 0], C_inf[0, 1], C_inf[1, 1]

        if self.eta0_mode == 'stationary_marginal':
            std = torch.sqrt(c22.clamp(min=1e-12))
            return std * torch.randn(shape, device=device)

        if self.eta0_mode == 'conditional':
            if x_0 is None:
                raise ValueError(
                    "eta0_mode='conditional' requires x_0 (shape [B,1,H,W])."
                )
            slope = c12 / c11.clamp(min=1e-12)
            cond_var = (c22 - c12 * c12 / c11.clamp(min=1e-12)).clamp(min=0)
            cond_std = torch.sqrt(cond_var + 1e-12)
            return slope * x_0 + cond_std * torch.randn_like(x_0)

        raise ValueError(f"Unknown eta0_mode: {self.eta0_mode}")

    # -------------------- correlated noise generation ------------------------

    def generate_correlated_noise(self, covariance_matrix, image_shape):
        """
        Generate correlated Gaussian noise for MNIST-shaped data.

        Args:
            covariance_matrix: [B, 2, 2] covariance matrices.
            image_shape: tuple containing at least (..., H, W).

        Returns:
            correlated_noise: [B, 2, H, W] samples from N(0, cov) per pixel.
        """
        batch_size = covariance_matrix.shape[0]
        height, width = image_shape[-2], image_shape[-1]

        L = torch.linalg.cholesky(covariance_matrix)        # [B, 2, 2]
        eps = torch.randn(batch_size, 2, height * width,
                          device=covariance_matrix.device)
        correlated = torch.bmm(L, eps)                       # [B, 2, H*W]
        return correlated.view(batch_size, 2, height, width)

    # -------------------- analytic forward-process pieces ---------------------

    def compute_exp_Mt(self, t):
        """Batched exp(M t). Returns [B, 2, 2]."""
        M = self.get_M()
        Mt = t.view(-1, 1, 1) * M.unsqueeze(0)
        return torch.linalg.matrix_exp(Mt)

    def compute_covariance(self, t):
        """
        Q(t) = integral_0^t exp(M s) Sigma exp(M^T s) ds   [B, 2, 2]

        Computed using Van Loan's formula:
            exp([[-M, Sigma], [0, M^T]] * t) = [[F1, F2], [0, F3]]
        with F1 = exp(-M t) and Q(t) = F1^{-1} F2.
        """
        M = self.get_M()
        device = M.device
        dtype = M.dtype

        G = torch.zeros(4, 4, device=device, dtype=dtype)
        G[:2, :2] = -M
        G[:2, 2:] = self.Sigma
        G[2:, 2:] = M.T

        Gt = t.view(-1, 1, 1) * G.unsqueeze(0)             # [B, 4, 4]
        expGt = torch.linalg.matrix_exp(Gt)

        F1 = expGt[:, :2, :2]                               # exp(-M t)
        F2 = expGt[:, :2, 2:]

        Q = torch.linalg.solve(F1, F2)                      # [B, 2, 2]
        Q = 0.5 * (Q + Q.transpose(-1, -2))                 # symmetrize
        eye = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
        Q = Q + 1e-6 * eye                                  # ridge
        return Q

    def compute_mean(self, x_0, eta_0, t):
        """E[x_t], E[eta_t] given (x_0, eta_0). Each [B, 1, H, W]."""
        E = self.compute_exp_Mt(t)                           # [B, 2, 2]
        a11 = E[:, 0, 0].view(-1, 1, 1, 1)
        a12 = E[:, 0, 1].view(-1, 1, 1, 1)
        a21 = E[:, 1, 0].view(-1, 1, 1, 1)
        a22 = E[:, 1, 1].view(-1, 1, 1, 1)
        mean_x = a11 * x_0 + a12 * eta_0
        mean_eta = a21 * x_0 + a22 * eta_0
        return mean_x, mean_eta

    def marginal_prob(self, x_0, eta_0, t):
        return self.compute_mean(x_0, eta_0, t), self.compute_covariance(t)

    # -------------------------------- training --------------------------------

    def forward(self, x, t=None, eps=None):
        """
        x has 2 channels: [image, eta_0_field].
        Returns (x_t, eta_t), eps_pred, eps, Q.
        """
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        x_0 = x[:, 0:1]
        eta_0 = x[:, 1:2]

        if t is None:
            t = 1e-3 + (self.total_time - 1e-3) * torch.rand(B, device=x.device)

        (mean_x, mean_eta), Q = self.marginal_prob(x_0, eta_0, t)

        L = torch.linalg.cholesky(Q)                        # [B, 2, 2]
        if eps is None:
            eps = torch.randn(B, 2, H, W, device=x.device)
        z_centered = torch.bmm(L, eps.view(B, 2, -1)).view(B, 2, H, W)

        x_t = mean_x + z_centered[:, 0:1]
        eta_t = mean_eta + z_centered[:, 1:2]

        eps_pred = self.model(torch.cat([x_t, eta_t], dim=1), t)
        return (x_t, eta_t), eps_pred, eps, Q

    def diffusion_loss(self, eps_pred, eps):
        """
        Full training loss:
            L = || eps_pred - eps ||^2
              + lambda_stab * stability_penalty(M)
              + lambda_l2   * ||M||_F^2      (skipped for fixed_norm mode)
        The penalty terms are included only when learn_M is True.
        """
        loss = torch.mean((eps_pred - eps) ** 2)
        if self.learn_M:
            loss = loss + self.stability_weight * self.stability_penalty()
            if self.M_constraint != 'fixed_norm':
                M = self.get_M()
                loss = loss + self.M_l2_weight * torch.sum(M ** 2)
        return loss

    # -------------------------------- sampling --------------------------------

    def _score_from_noise(self, eps_pred, Q):
        """score = -L^{-T} eps_pred,  L = chol(Q)."""
        B, _, H, W = eps_pred.shape
        L = torch.linalg.cholesky(Q)
        eps_flat = eps_pred.view(B, 2, -1)
        s = torch.linalg.solve_triangular(
            L.transpose(-1, -2), eps_flat, upper=True
        )
        return -s.view(B, 2, H, W)

    @torch.no_grad()
    def _reverse_step(self, x_eta, t_tensor):
        """One Euler-Maruyama step of the reverse-time SDE."""
        eps_pred = self.model(x_eta, t_tensor)
        Q = self.compute_covariance(t_tensor)
        score = self._score_from_noise(eps_pred, Q)

        x = x_eta[:, 0:1]
        eta = x_eta[:, 1:2]
        sx = score[:, 0:1]
        se = score[:, 1:2]

        M = self.get_M()
        D = self.D
        S = self.Sigma
        dt = self.dt
        sdt = math.sqrt(dt)

        # Reverse drift: -M z + Sigma * score
        drift_x = -(M[0, 0] * x + M[0, 1] * eta) + S[0, 0] * sx + S[0, 1] * se
        drift_e = -(M[1, 0] * x + M[1, 1] * eta) + S[1, 0] * sx + S[1, 1] * se

        # Reverse-time Wiener increments via D
        n1 = torch.randn_like(x)
        n2 = torch.randn_like(eta)
        nx = D[0, 0] * n1 + D[0, 1] * n2
        ne = D[1, 0] * n1 + D[1, 1] * n2

        x_new = x + dt * drift_x + sdt * nx
        e_new = eta + dt * drift_e + sdt * ne
        return torch.cat([x_new, e_new], dim=1)

    @torch.no_grad()
    def sampling(self, n_samples, device="cuda"):
        """
        Initialize z(T) from the true stationary distribution N(0, C_inf),
        then integrate the reverse SDE down to t ~ 0.
        """
        C_inf = self.compute_C_inf()
        C_inf_batch = C_inf.unsqueeze(0).expand(n_samples, -1, -1)
        x_eta = self.generate_correlated_noise(
            C_inf_batch, (self.image_size, self.image_size)
        )

        for step in tqdm(range(self.timesteps - 2, 0, -1), desc="Sampling"):
            t_tensor = (step * self.dt) * torch.ones(n_samples, device=device)
            x_eta = self._reverse_step(x_eta, t_tensor)

        x_final = x_eta[:, 0:1].clip(-1, 1)
        x_final = (x_final + 1.0) / 2.0
        return x_final.detach(), x_eta[:, 1:2].detach()

    @torch.no_grad()
    def sampling_from_intermediate(self, x_eta_t, t_step, device="cuda"):
        n_samples = x_eta_t.shape[0]
        x_eta = x_eta_t.clone()
        for step in tqdm(range(t_step, 0, -1), desc="Sampling from intermediate"):
            t_tensor = (step * self.dt) * torch.ones(n_samples, device=device)
            x_eta = self._reverse_step(x_eta, t_tensor)
        x_final = x_eta[:, 0:1].clip(-1, 1)
        x_final = (x_final + 1.0) / 2.0
        return x_final.detach(), x_eta[:, 1:2].detach()