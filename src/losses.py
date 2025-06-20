# src/losses.py
import torch
import torch.nn.functional as F

def get_loss_fn(noise, graph, train):
    """
    Returns the appropriate loss function for training or validation.
    """
    if train:
        return DWDSE(noise, graph)
    else:
        # For validation, we can use a simplified loss for now.
        # A proper implementation would estimate the likelihood.
        return DWDSE(noise, graph)


class DWDSE:
    """
    Implements the Denoising Score Entropy (DWDSE) loss for training.
    This version operates in the embedding space.
    """
    def __init__(self, noise, graph):
        self.noise = noise
        self.graph = graph

    def __call__(self, model, x_0_ids):
        # x_0_ids is the input tensor of discrete token IDs.
        
        # 1. Get the initial embeddings from the discrete token IDs.
        # The model's backbone (Mamba) has the embedding table.
        with torch.no_grad():
            x_0_embed = model.mamba.embedding(x_0_ids).detach()

        # 2. Sample a random timestep 't' for each item in the batch
        t = torch.randint(0, self.noise.num_timesteps, (x_0_embed.shape[0],), device=x_0_embed.device)
        
        # 3. Add noise to the initial *embeddings* to get x_t
        x_t_embed, noise_added = self.noise.sample(x_0_embed, t)
        
        # 4. Get the model's prediction of the noise
        predicted_noise = model(x_t_embed, t)

        # 5. Calculate the weighted mean squared error between the actual noise
        #    and the predicted noise.
        lmbda = self.graph.get_lambda(t, predicted_noise.device)
        
        # Ensure dimensions match for loss calculation
        if predicted_noise.shape != noise_added.shape:
             raise ValueError(f"Shape mismatch between predicted noise {predicted_noise.shape} and added noise {noise_added.shape}")

        mse = F.mse_loss(predicted_noise.flatten(1), noise_added.flatten(1), reduction='none').mean(1)
        
        loss = lmbda * mse
        return loss

class L_DWDSE:
    """
    Placeholder for a validation loss. For now, we can reuse the training loss logic.
    """
    def __call__(self, model, x_0):
        # This can be properly implemented later.
        # For now, let's just return a placeholder or use the training loss.
        return torch.tensor(0.0)