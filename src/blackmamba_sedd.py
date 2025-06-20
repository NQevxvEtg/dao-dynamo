# src/blackmamba_sedd.py
import torch
import torch.nn as nn
import math

from mamba_model import MambaModel


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class BlackMambaSEDD(nn.Module):
    """
    A wrapper for a MambaModel backbone that operates in the embedding space.
    It takes noisy embeddings and predicts the noise.
    """
    def __init__(self, mamba_model: MambaModel, d_model: int, time_embedding_dim: int, **kwargs):
        super().__init__()
        self.mamba = mamba_model 

        # The Mamba backbone's decoder is what we'll use.
        self.decoder = self.mamba.decoder
        
        mamba_hidden_size = self.mamba.config.hidden_size

        # The model's input is now the embedding dim, not 1.
        self.input_proj = nn.Linear(d_model, mamba_hidden_size)

        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embedding_dim, mamba_hidden_size),
            nn.SiLU(),
            nn.Linear(mamba_hidden_size, mamba_hidden_size),
        )
        
        # The output head now predicts noise, so its output dim must match the embedding dim.
        self.output_proj = nn.Linear(mamba_hidden_size, d_model)


    def forward(self, x_embed: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Accepts a noisy embedding 'x_embed' and timestep 't', and predicts the noise.
        """
        # 1. Project the input embedding into the model's hidden dimension
        hidden_states = self.input_proj(x_embed)
        
        # 2. Add Time Conditioning
        time_embeds = self.time_embed(t)
        time_embeds_proj = self.time_proj(time_embeds)
        hidden_states = hidden_states + time_embeds_proj.unsqueeze(1)
        
        # 3. Pass through the Mamba decoder backbone
        hidden_states = self.decoder(hidden_states)
        
        # 4. Project the final hidden state to predict the noise
        predicted_noise = self.output_proj(hidden_states)
        
        return predicted_noise