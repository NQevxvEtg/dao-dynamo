# src/graph_lib.py
import torch
import numpy as np

def get_graph(graph_name, noise_object, device):
    """
    Factory function to get the appropriate graph based on config.
    Now accepts the actual noise object instead of a config object.
    """
    if graph_name == 'absorb':
        return Absorb(noise_object, device)
    else:
        raise ValueError(f"Unknown graph type: {graph_name}")

class Absorb:
    """
    Implements the absorbing state graph for the diffusion process.
    """
    def __init__(self, noise_object, device):
        self.noise = noise_object
        self.device = device
        self.num_timesteps = self.noise.num_timesteps
        self.lambda_schedule = self._setup_lambda_schedule()

    def _setup_lambda_schedule(self):
        """
        Sets up the lambda schedule, which controls the weighting in the loss function.
        This is derived from the noise schedule (alpha).
        """
        alphas_cumprod = self.noise.alphas_cumprod
        
        lmbda = (self.noise.alphas ** 2) / (2 * (self.noise.sigma_hat ** 2) + 1e-8)
        
        # This will now be a 0-dim tensor from the fix in noise_lib.py
        lmbda_0_val = 1. / (2. * (self.noise.sigma_hat_0 ** 2))
        
        # --- THE FIX ---
        # Make the value 1D and ensure it's float32 for concatenation.
        lmbda_0 = lmbda_0_val.unsqueeze(0).to(dtype=torch.float32)
        # --- END FIX ---
        
        full_lambda_schedule = torch.cat([lmbda_0, lmbda], dim=0)
        
        return full_lambda_schedule.to(self.device)

    def get_lambda(self, t, device):
        """
        Retrieves the lambda value for a given timestep 't'.
        """
        return self.lambda_schedule[t + 1].to(device)