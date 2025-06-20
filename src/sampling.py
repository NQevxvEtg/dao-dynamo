# src/sampling.py
import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_pc_sampler(graph, noise, shape, sampler_name, steps, device):
    """
    Factory function for getting the Predictor-Corrector sampler.
    Operates in the embedding space.
    """
    def sampler(model):
        # The actual sampling function that will be returned.
        # 'model' here is the BlackMambaSEDD object.
        predictor = get_predictor(sampler_name)
        corrector = get_corrector(sampler_name)
        
        # Start with random noise in the embedding space
        x = torch.randn(shape, device=device)

        # Iteratively denoise the data using the predictor-corrector steps
        for i in tqdm(reversed(range(steps)), desc=f'{sampler_name} sampler', total=steps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            # Apply corrector then predictor
            x = corrector(x, t, model, graph)
            x = predictor(x, t, model, graph)
        
        # Returns the final denoised embedding
        return x
    return sampler

def get_predictor(sampler_name):
    if sampler_name == 'analytic':
        return AnalyticPredictor
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

def get_corrector(sampler_name):
    if sampler_name == 'analytic':
        return AnalyticCorrector
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


# --- Predictor-Corrector Implementations ---
# These now operate on float tensors (embeddings)

def AnalyticPredictor(x, t, model, graph):
    if t.sum() == 0:
        return x
    
    # The model now predicts the noise directly
    predicted_noise = model(x, t)
    
    # Predictor step formula
    x = (1. / graph.noise.sqrt_alphas_cumprod[t])[:, None, None] * (
        x - graph.noise.sqrt_one_minus_alphas_cumprod[t][:, None, None] * predicted_noise
    )
    return x

def AnalyticCorrector(x, t, model, graph):
    if t.sum() == 0:
        return x
    
    # The model predicts the noise
    predicted_noise = model(x, t)
    
    # Corrector step formula
    x = (
        graph.noise.sqrt_one_minus_alphas_cumprod[t][:, None, None] * (-predicted_noise)
        + graph.noise.sqrt_alphas_cumprod[t][:, None] * x
    )
    return x