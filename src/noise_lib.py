# src/noise_lib.py
import torch
import torch.nn.functional as F
import numpy as np

def get_noise(config, device):
    """
    Factory function to get the appropriate noise schedule based on config.
    """
    if config.noise == 'loglinear':
        return LogLinear(num_timesteps=config.num_timesteps, device=device)
    else:
        raise ValueError(f"Unknown noise type: {config.noise}")

class LogLinear:
    """
    Implements a logarithmic-linear noise schedule. This schedule determines
    how much noise is added at each step of the diffusion process.
    """
    def __init__(self, num_timesteps=1000, start_t=0.0001, end_t=0.9999, device='cpu'):
        self.num_timesteps = num_timesteps
        
        self.betas = np.linspace(start_t, end_t, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        
        self.betas = torch.tensor(self.betas, dtype=torch.float32, device=device)
        self.alphas = torch.tensor(self.alphas, dtype=torch.float32, device=device)
        self.alphas_cumprod = torch.tensor(self.alphas_cumprod, dtype=torch.float32, device=device)
        
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # --- THE FIX ---
        # Perform the calculation entirely in PyTorch to keep it on the correct device
        # and ensure it remains a float32 tensor, preventing numpy's float64 conversion.
        self.sigma_hat_0 = torch.sqrt(1. - self.alphas_cumprod[0])
        # --- END FIX ---
        
        self.sigma_hat = torch.sqrt(1. - self.alphas_cumprod)

    def sample(self, x_0, t):
        """
        Samples a noisy version of x_0 at a given timestep t.
        This is the forward diffusion process q(x_t | x_0).
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        noise = torch.randn_like(x_0)
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise


def extract(a, t, x_shape):
    """
    Helper function to extract the correct values from a schedule for a batch of timesteps.
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))