# dao-dynamo

`dao-dynamo` is an experimental AI agent designed for continuous learning and self-cultivation. It's built on a novel architecture that merges a state-of-the-art State-Space Model (BlackMamba) with a discrete diffusion generative process (SEDD), all orchestrated by a unique cognitive framework that simulates internal states like focus, curiosity, and self-reflection.

Unlike static, pre-trained models, `dao-dynamo` is designed to be a "living" system that learns from every interaction, reflects on its internal state, and dynamically adapts its own parameters over time.

## Core Architecture

The project is a synthesis of three key concepts, resulting in a unique and powerful generative agent:

1.  **Cognitive Architecture (The DAO Framework):** The agent's "mind" is governed by a homeostatic framework that provides metacognitive functions.

      * **Emotional Core & Heart:** A regulatory system that generates cognitive rhythms and modulates learning parameters like focus and curiosity based on long-term performance. The `Heart` module performs periodic "rebase" adjustments, allowing the model to automatically tune itself.
      * **Self-Reflection & Prompting:** The agent assesses its own confidence and internal consistency and generates its own introspective prompts to guide its "thought" process.

2.  **Generative Model (Score Entropy Discrete Diffusion - SEDD):** The core generative paradigm is based on the principles from the "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" paper.

      * Instead of traditional autoregressive prediction, `dao-dynamo` learns to reverse a noising process on discrete data.
      * It operates in the **embedding space**, learning to denoise embeddings using the Denoising Score Entropy (`DWDSE`) loss function.
      * Generation is handled by a predictor-corrector sampler that synthesizes new embeddings from noise.

3.  **Sequence Backbone (BlackMamba):** The engine of the diffusion model is **BlackMamba**, a powerful State-Space Model (SSM).

      * This replaces the more common Transformer/U-Net architecture in diffusion models, leveraging the Mamba architecture's efficiency and linear-time scaling for long sequences.
      * The result is the `BlackMambaSEDD` model, a unique synthesis that uses a Mamba model as a time-aware denoising engine within the SEDD framework.

## Current Status & Known Issues

The primary architectural refactoring is complete. The `BlackMambaSEDD` model is fully integrated as the core of the `ContinuouslyReasoningPredictor`, and the surrounding cognitive framework is operational. The project is currently in an integration, training, and validation phase.

**Known Issue:** There is a suspected low-level bug related to the custom CUDA operations for the BlackMamba backbone. The `docker-compose.yml` file is configured to run the backend with the `gdb` debugger to help isolate a potential segmentation fault in the patched `selective_scan_interface.py` C++/CUDA code. This is the top-priority issue to be resolved.

## Roadmap

The following tasks are pending to make the new architecture fully stable and operational:

  * [ ] **Debug CUDA Operations**: Resolve the low-level bug in the patched `selective_scan_interface.py` to ensure training stability.
  * [ ] **Validate Dynamic Adjustments**: Systematically test and refine the `Heart` module's rebase logic to ensure it effectively and automatically optimizes the model's core parameters over long-term training.
  * [ ] **Formalize Configuration**: Move model and SEDD parameters from `main.py` to a formal configuration system (e.g., Hydra, YAML files) for better experiment management.
  * [ ] **Implement Validation Loop**: The validation logic in `training_manager.py` must be updated to use a proper likelihood-bound calculation (`L_DWDSE`) as described in the SEDD paper.
  * [ ] **Enhance Self-Reflection**: Fully integrate the `SelfReflectionModule` by robustly converting generated token sequences into meaningful embeddings for reflection.
  * [ ] **Review Tokenizer**: Evaluate if the current `bert-base-uncased` tokenizer is optimal, or if another tokenizer would be more suitable.
  * [ ] **Scale Up**: Once stable, begin scaled-up training runs on the full dataset to properly evaluate the model's capabilities.

## Getting Started

1.  **Environment**: Ensure you have Docker and Docker Compose installed. You will also need an NVIDIA GPU with the appropriate drivers and CUDA toolkit installed.
2.  **Configuration**: Create a `.env` file in the root directory for database credentials and other environment variables.
3.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
4.  **Access**:
      * **Backend API (Swagger UI)**: `http://localhost:8000/docs`
      * **Frontend UI**: `http://localhost:3000`