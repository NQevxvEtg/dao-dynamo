# dao/src/main.py
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import logging
import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer
from sqlalchemy.orm import Session
from sqlalchemy import text
import io
import csv
from collections import deque
import asyncio
from datetime import datetime
import json
import uuid
from omegaconf import OmegaConf, DictConfig

from src.api_models import GenerateRequest, GenerateResponse, InternalThoughtResponse, TrainingStatusResponse
from src.data_processing import (
    TextDataset, collate_fn, load_texts_from_directory,
    create_file_manifest, get_changed_files, load_texts_from_specific_files
)
from torch.utils.data import DataLoader

from src.database import init_db, get_db
from src.model_architecture import ContinuouslyReasoningPredictor
from src.utils import Vocabulary, decode_sequence
from src.websocket_manager import ConnectionManager, broadcast_state_update

from src.training_manager import (
    TrainingState, run_model_training,
    DATA_DIR, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, MAX_SEQ_LEN, SAVE_INTERVAL_BATCHES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="dao - The Living Dynamo",
    description="A continuously learning and self-cultivating AI agent.",
    version="0.2.1",
)

MODELS_DIR = "./models"
MODEL_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "dao_model_checkpoint.pth")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
MANIFEST_PATH = os.path.join(MODELS_DIR, "vocab_manifest.json")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INTERNAL_THOUGHT_INTERVAL_SECONDS = 10
SILENT_THOUGHT_PHASE_COUNT = 3

training_state = TrainingState()
websocket_manager = ConnectionManager()


async def _generate_and_store_internal_thought():
    while True:
        await asyncio.sleep(INTERNAL_THOUGHT_INTERVAL_SECONDS)
        try:
            if training_state.is_training_active:
                continue
            async with app.state.model_lock:
                model = app.state.model
                vocab = app.state.vocab
                
                self_reflection_prompt = "" # Let generate_text handle the prompt generation
                if app.state.internal_thought_sequence_num >= SILENT_THOUGHT_PHASE_COUNT:
                    current_states = {
                        "confidence": model.latest_confidence.mean().item() if model.latest_confidence is not None else 0.0,
                        "meta_error": model.latest_meta_error.mean().item() if model.latest_meta_error is not None else 0.0,
                        "focus": model.emotions.get_focus(),
                        "curiosity": model.emotions.get_curiosity()
                    }
                    self_reflection_prompt = model.self_prompting_module.generate_prompt(current_states)

                with torch.no_grad():
                    thought_text, confidence, meta_error, focus, curiosity, prompt_text_used = \
                        await model.generate_text(vocab, self_reflection_prompt, max_len=64)
                    
                    await model.perform_self_reflection(thought_text)

                thought_entry = InternalThoughtResponse(
                    thought=thought_text, timestamp=datetime.now(), confidence=confidence,
                    meta_error=meta_error, focus=focus, curiosity=curiosity, prompt_text=prompt_text_used
                )
                app.state.internal_thoughts_queue.append(thought_entry)
                app.state.internal_thought_sequence_num += 1

                await broadcast_state_update(
                    manager=websocket_manager,
                    model=model,
                    is_training=False,
                    message="State update after internal thought."
                )

        except Exception as e:
            logger.error(f"Error in internal thought generation: {e}", exc_info=True)

def reset_optimizer(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Training task failed with exception: {exc}", exc_info=exc)
    finally:
        logger.info("Training task finished. Resetting optimizer state for interactive mode.")
        model = app.state.model
        app.state.optimizer = torch.optim.Adam(model.parameters(), lr=model.emotions.get_curiosity())
        app.state.scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
        logger.info("Optimizer and GradScaler have been reset.")


@app.on_event("startup")
async def startup_event():
    logger.info(f"dao is igniting... setting up core systems on device: {DEVICE}")
    app.state.model_lock = asyncio.Lock()
    app.state.session_id = str(uuid.uuid4())
    await init_db()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = Vocabulary(tokenizer)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if os.path.exists(VOCAB_PATH):
        vocab.load_vocab(VOCAB_PATH)
    else:
        logger.warning(f"Vocabulary file not found at {VOCAB_PATH}.")
        try:
            train_data_dir = os.path.join(DATA_DIR, 'train')
            all_texts = load_texts_from_directory(train_data_dir)
            if all_texts:
                vocab.build_vocab(all_texts)
                vocab.save_vocab(VOCAB_PATH)
        except Exception as e:
            logger.error(f"Error building vocabulary from training data: {e}")
            vocab = Vocabulary(tokenizer)

    app.state.vocab = vocab
    
    config = OmegaConf.create({
        "model": {
            "pretrained_model_name": None,
            "d_model": 768,
            "n_layer": 12,
            "time_embedding_dim": 256,
            "sentence_transformer_model": "BAAI/bge-base-en-v1.5",
            "max_seq_len": MAX_SEQ_LEN
        },
        "sedd": {
            "graph": "absorb",
            "noise": "loglinear",
            "sampler": "analytic",
            "num_timesteps": 1000,
        }
    })

    model = ContinuouslyReasoningPredictor(
        vocab_size=vocab.vocab_size, 
        sos_token_id=vocab.sos_token_id,
        eos_token_id=vocab.eos_token_id,
        pad_token_id=vocab.pad_token_id,
        device=DEVICE,
        config=config
    )
    
    app.state.model = model
    # Move the entire model structure to the designated device (GPU)
    app.state.model.to(DEVICE)

    app.state.optimizer = torch.optim.Adam(model.parameters(), lr=model.emotions.get_curiosity())

    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
            app.state.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                 app.state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model checkpoint loaded from {MODEL_CHECKPOINT_PATH}.")
        except Exception as e:
            logger.warning(f"Could not load model checkpoint: {e}. Starting with fresh weights.")
    else:
        logger.warning("Model checkpoint not found. Model will use from-scratch weights.")
    
    app.state.scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    app.state.internal_thoughts_queue = deque(maxlen=50)
    app.state.internal_thought_sequence_num = 0
    
    asyncio.create_task(_generate_and_store_internal_thought())
    logger.info("dao's initial core systems are online.")


@app.post("/generate_response", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest, db: Session = Depends(get_db)):
    if training_state.is_training_active:
        raise HTTPException(status_code=400, detail="Chat paused during training.")

    model = app.state.model
    vocab = app.state.vocab
    optimizer = app.state.optimizer
    scaler = app.state.scaler
    response_text = ""
    continuous_learning_loss_value = 0.0

    try:
        async with app.state.model_lock:
            model.train()
            
            user_tokens = vocab.encode(request.prompt)
            target_ids = user_tokens[:MAX_SEQ_LEN]
            target_ids_padded = target_ids + [vocab.pad_token_id] * (MAX_SEQ_LEN - len(target_ids))
            target_tensor = torch.tensor([target_ids_padded], dtype=torch.long, device=DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            loss = await model.learn_one_step(target_tensor)
            
            if torch.is_tensor(loss) and loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                continuous_learning_loss_value = loss.item()

            model.eval()
            with torch.no_grad():
                response_text, _, _, _, _, _ = await model.generate_text(vocab, request.prompt, max_len=request.max_length)
                await model.perform_self_reflection(response_text)

            await broadcast_state_update(
                manager=websocket_manager, model=model, is_training=False,
                message="State update after interactive response.",
                continuous_learning_loss=continuous_learning_loss_value
            )

    except Exception as e:
        logger.error(f"Error in response pipeline: {e}", exc_info=True)
        response_text = f"Error: An internal exception occurred. ({e})"
    
    confidence = model.latest_confidence.mean().item() if model.latest_confidence is not None else 0.0
    meta_error = model.latest_meta_error.mean().item() if model.latest_meta_error is not None else 0.0
    
    try:
        db.execute(text("""
            INSERT INTO chat_messages (session_id, sender, message_text)
            VALUES (:session_id, 'user', :message_text)
        """), {"session_id": app.state.session_id, "message_text": request.prompt})

        db.execute(text("""
            INSERT INTO chat_messages (session_id, sender, message_text, confidence, meta_error, focus, curiosity)
            VALUES (:session_id, 'dao', :message_text, :confidence, :meta_error, :focus, :curiosity)
        """), {
            "session_id": app.state.session_id, "message_text": response_text,
            "confidence": confidence, "meta_error": meta_error,
            "focus": model.emotions.get_focus(), "curiosity": model.emotions.get_curiosity()
        })
        db.commit()
    except Exception as e:
        logger.error(f"DB error saving chat message: {e}")
        db.rollback()

    return GenerateResponse(
        response=response_text,
        confidence=confidence,
        meta_error=meta_error,
        focus=model.emotions.get_focus(),
        curiosity=model.emotions.get_curiosity(),
        continuous_learning_loss=continuous_learning_loss_value
    )

@app.get("/internal_thought", response_model=list[InternalThoughtResponse], summary="Get dao's internal thoughts")
async def get_internal_thoughts():
    return list(app.state.internal_thoughts_queue)


@app.post("/start_training", response_model=TrainingStatusResponse, summary="Start model training")
async def start_training():
    if training_state.is_training_active:
        raise HTTPException(status_code=400, detail="Training is already active.")

    try:
        task = asyncio.create_task(run_model_training(
            model=app.state.model,
            vocab=app.state.vocab,
            optimizer=app.state.optimizer,
            training_state=training_state,
            websocket_manager=websocket_manager,
            model_checkpoint_path=MODEL_CHECKPOINT_PATH,
            device=DEVICE,
            scaler=app.state.scaler,
            lock=app.state.model_lock
        ))
        task.add_done_callback(reset_optimizer)
        logger.info("Training task initiated via API.")
        return TrainingStatusResponse(is_training_active=True, message="Training started successfully.")
    except Exception as e:
        logger.error(f"Error initiating training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate training task: {e}")

# ... (rest of the file is unchanged, but included for completeness) ...

@app.post("/stop_training", response_model=TrainingStatusResponse, summary="Stop model training")
async def stop_training():
    if not training_state.is_training_active:
        return TrainingStatusResponse(is_training_active=False, message="Training not active to stop.")

    training_state.stop_training_flag.set()
    logger.info("Training stop signal sent.")
    
    asyncio.create_task(broadcast_state_update(
        manager=websocket_manager,
        model=app.state.model,
        is_training=False,
        message="Training stop signal sent. Ceasing shortly."
    ))
    
    return TrainingStatusResponse(
        is_training_active=False,
        message="Training stop signal sent. Ceasing shortly."
    )


@app.get("/training_status", response_model=TrainingStatusResponse, summary="Get current training status")
async def get_training_status():
    model = app.state.model
    return TrainingStatusResponse(
        is_training_active=training_state.is_training_active,
        current_epoch=training_state.current_epoch,
        current_batch=training_state.current_batch,
        total_batches_in_epoch=training_state.total_batches_in_epoch,
        train_loss=training_state.train_loss,
        val_loss=training_state.val_loss if training_state.val_loss != float('inf') else None,
        best_val_loss=training_state.best_val_loss if training_state.best_val_loss != float('inf') else None,
        confidence=model.latest_confidence.mean().item() if model.latest_confidence is not None else 0.0,
        meta_error=model.latest_meta_error.mean().item() if model.latest_meta_error is not None else 0.0,
        focus=model.emotions.get_focus(),
        curiosity=model.emotions.get_curiosity(),
        message="Training active." if training_state.is_training_active else "Training not active."
    )

@app.websocket("/ws/training_updates")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        logger.info(f"WebSocket {websocket.client} disconnected.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket connection for {websocket.client}: {e}")
    finally:
        websocket_manager.disconnect(websocket)


@app.get("/export_chat", summary="Export chat messages to CSV")
async def export_chat(db: Session = Depends(get_db)):
    try:
        query = text("SELECT timestamp, session_id, sender, message_text, confidence, meta_error, focus, curiosity FROM chat_messages ORDER BY timestamp;")
        result = db.execute(query).fetchall()

        if not result:
            return JSONResponse(status_code=404, content={"message": "No chat history found to export."})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(result[0]._fields)
        writer.writerows(result)
        output.seek(0)
        headers = {"Content-Disposition": "attachment; filename=dao_chat_history.csv"}
        return StreamingResponse(output, headers=headers)
    except Exception as e:
        logger.error(f"Error exporting chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export chat: {e}")

@app.delete("/clear_chat_history", summary="Clear all chat messages from the database")
async def clear_chat_history(db: Session = Depends(get_db)):
    try:
        db.execute(text("TRUNCATE TABLE chat_messages;"))
        db.commit()
        logger.info("Chat messages table truncated successfully.")
        return JSONResponse(status_code=200, content={"message": "Chat history cleared successfully."})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {e}")

@app.get("/cognitive_state_history", summary="Get all saved cognitive state history")
async def get_cognitive_state_history(db: Session = Depends(get_db)):
    try:
        query = text("SELECT * FROM cognitive_state_history ORDER BY timestamp;")
        result = db.execute(query).mappings().all()
        return result
    except Exception as e:
        logger.error(f"Error fetching cognitive state history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch cognitive state history.")

@app.get("/export_cognitive_state", summary="Export cognitive state history to CSV")
async def export_cognitive_state(db: Session = Depends(get_db)):
    try:
        query = text("SELECT * FROM cognitive_state_history ORDER BY timestamp;")
        result = db.execute(query).fetchall()
        
        if not result:
            return JSONResponse(status_code=404, content={"message": "No cognitive state history found to export."})

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(result[0]._fields)
        writer.writerows(result)
        output.seek(0)
        headers = {"Content-Disposition": "attachment; filename=dao_cognitive_state_history.csv"}
        return StreamingResponse(output, headers=headers)
    except Exception as e:
        logger.error(f"Error exporting cognitive state history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export cognitive state history: {e}")

@app.delete("/clear_cognitive_state", summary="Clear all cognitive state history")
async def clear_cognitive_state(db: Session = Depends(get_db)):
    try:
        db.execute(text("TRUNCATE TABLE cognitive_state_history;"))
        db.commit()
        logger.info("Cognitive state history table truncated successfully.")
        return JSONResponse(status_code=200, content={"message": "Cognitive state history cleared successfully."})
    except Exception as e:
        logger.error(f"Error clearing cognitive state history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cognitive state history: {e}")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")