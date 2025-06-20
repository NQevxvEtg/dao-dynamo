# dao/src/model_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import asyncio
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from .utils import Vocabulary, initialize_weights, decode_sequence
from .emotion import EmotionalCore
from .heart import Heart
from .self_reflection import SelfReflectionModule
from .self_prompting import SelfPromptingModule
from .blackmamba_sedd import BlackMambaSEDD
from .losses import get_loss_fn
from . import graph_lib
from . import noise_lib
from . import sampling

from mamba_model import MambaModel
from mamba_config import MambaConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ContinuouslyReasoningPredictor(nn.Module):
    def __init__(self, vocab_size: int, sos_token_id: int, eos_token_id: int, pad_token_id: int, device: torch.device, config: DictConfig):
        super().__init__()
        self.device = device
        self.config = config
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id

        # --- Mamba Backbone Initialization ---
        mamba_config = MambaConfig(
            hidden_size=config.model.d_model,
            num_layers=config.model.n_layer,
            vocab_size=vocab_size,
            conv_dimension=4,
            state_size=16
        )
        loaded_mamba = MambaModel(
            config=mamba_config, 
            max_sequence_length=config.model.max_seq_len,
            post_process=False
        )
        logger.info(f"New BlackMamba backbone initialized.")
        
        # --- SEDD Model Initialization ---
        self.sedd_model = BlackMambaSEDD(
            mamba_model=loaded_mamba,
            d_model=config.model.d_model,
            time_embedding_dim=config.model.time_embedding_dim
        ) 

        # --- Other Cognitive Modules ---
        self.embedding_model = SentenceTransformer(config.model.sentence_transformer_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.reflection_input_projection = nn.Linear(self.embedding_dim, config.model.d_model)
        initialize_weights(self.reflection_input_projection)
        
        self.emotions = EmotionalCore()
        self.heart = Heart()
        self.self_reflection_module = SelfReflectionModule(
            model_dims=config.model.d_model,
            reflection_dims=config.model.d_model
        )
        self.self_prompting_module = SelfPromptingModule()
        
        # --- Diffusion Components ---
        self.noise = noise_lib.get_noise(self.config.sedd, self.device)
        self.graph = graph_lib.get_graph(self.config.sedd.graph, self.noise, self.device)
        self.sedd_train_loss_fn = get_loss_fn(self.noise, self.graph, train=True)
        self.sedd_val_loss_fn = get_loss_fn(self.noise, self.graph, train=False)

        # --- Internal States ---
        self.slow_state = nn.Parameter(torch.randn(1, config.model.d_model) * 0.01)
        self.fast_state = nn.Parameter(torch.zeros(1, config.model.d_model), requires_grad=False)
        self.latest_confidence = torch.tensor([0.5])
        self.latest_meta_error = torch.tensor([0.0])
        self.latest_state_drift = 0.0
        self.latest_heart_metrics = {}

        logger.info("ContinuouslyReasoningPredictor initialized with BlackMambaSEDD core.")

    def _decode_embedding_to_ids(self, final_embedding: torch.Tensor) -> torch.Tensor:
        """
        Finds the closest token in the vocabulary for each position in the denoised embedding.
        """
        # Get the vocabulary embeddings from the backbone model
        vocab_embeddings = self.sedd_model.mamba.embedding.weight.data.T.unsqueeze(0)
        
        # Normalize embeddings for cosine similarity
        final_embedding_norm = F.normalize(final_embedding, p=2, dim=-1)
        vocab_embeddings_norm = F.normalize(vocab_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        # (B, L, D) @ (B, D, V) -> (B, L, V)
        cosine_sim = torch.matmul(final_embedding_norm, vocab_embeddings_norm)
        
        # Find the index of the highest similarity for each position
        _, nearest_token_ids = torch.max(cosine_sim, dim=-1)
        
        return nearest_token_ids

    async def _get_predicted_sequence(self, batch_size: int, sequence_length: int, stop_event: asyncio.Event = None) -> torch.Tensor:
        self.emotions.update()
        sampling_steps = self.emotions.get_focus()
        logger.debug(f"Starting SEDD sampling with {sampling_steps} steps.")
        
        # The shape for the sampler is now (batch, seq_len, embedding_dim)
        shape = (batch_size, sequence_length, self.config.model.d_model)

        sampling_fn = sampling.get_pc_sampler(
            self.graph, self.noise, 
            shape, 
            self.config.sedd.sampler,
            steps=sampling_steps,
            device=self.device
        )
        with torch.no_grad():
            # The sampler now returns a denoised embedding
            denoised_embedding = await asyncio.to_thread(sampling_fn, self.sedd_model)
            # We must decode this embedding to get token IDs
            generated_ids_tensor = self._decode_embedding_to_ids(denoised_embedding)

        await asyncio.sleep(0)
        return generated_ids_tensor

    async def learn_one_step(self, x_0: torch.Tensor, stop_event: asyncio.Event = None) -> torch.Tensor:
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Training stopped by user.")
        
        # The loss function now handles the embedding lookup internally
        loss = self.sedd_train_loss_fn(self.sedd_model, x_0)
        return loss.mean()

    async def validate_one_step(self, x_0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            loss = self.sedd_val_loss_fn(self.sedd_model, x_0)
        return loss.mean()

    async def generate_text(self, vocab: Vocabulary, input_prompt: str, max_len: int = 256) -> tuple:
        self.eval()
        with torch.no_grad():
            self_reflection_prompt = input_prompt or self.self_prompting_module.generate_prompt({})
            generated_ids_tensor = await self._get_predicted_sequence(batch_size=1, sequence_length=max_len)
            thought_text = decode_sequence(generated_ids_tensor.squeeze(0).tolist(), vocab, self.eos_token_id)
            
            confidence = self.latest_confidence.mean().item() if self.latest_confidence is not None else 0.0
            meta_error = self.latest_meta_error.mean().item() if self.latest_meta_error is not None else 0.0
            focus = self.emotions.get_focus()
            curiosity = self.emotions.get_curiosity()

            logger.info(f"Generated Thought: '{thought_text}'")
            
            return thought_text, confidence, meta_error, focus, curiosity, self_reflection_prompt

    async def perform_self_reflection(self, text: str):
        self.eval()
        with torch.no_grad():
            self.emotions.update()

            reasoned_embedding = self.embedding_model.encode([text], convert_to_tensor=True, device=self.device)
            reasoned_state = self.reflection_input_projection(reasoned_embedding)

            batch_size = reasoned_state.shape[0]
            current_confidence, current_meta_error = self.self_reflection_module(
                reasoned_state=reasoned_state,
                fast_state=self.fast_state.expand(batch_size, -1),
                slow_state=self.slow_state.expand(batch_size, -1)
            )

            self.latest_confidence = current_confidence
            self.latest_meta_error = current_meta_error
            similarity = F.cosine_similarity(self.slow_state, self.fast_state.mean(dim=0, keepdim=True))
            self.latest_state_drift = (1.0 - similarity.mean()).item()
            self.latest_heart_metrics = self.heart.beat(self.emotions, self.latest_confidence, self.latest_meta_error)
            
            new_fast_state = self.fast_state + 0.1 * (reasoned_state - self.fast_state)
            self.fast_state.copy_(new_fast_state.mean(dim=0, keepdim=True))
            
            self.slow_state.data += self.emotions.get_curiosity() * (self.fast_state - self.slow_state)

            logger.info(f"Reflection complete. Confidence: {self.latest_confidence.mean().item():.4f}, Meta-Error: {self.latest_meta_error.mean().item():.4f}")