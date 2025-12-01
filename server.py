import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
from whisper_streamer.core import WhisperStreamer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time
import json
import os
import asyncio

app = FastAPI()

# Load model globally to save memory
MODEL_ID = "openai/whisper-tiny.en"
DEVICE = "cpu"
print(f"Loading global model {MODEL_ID}...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
model.config.forced_decoder_ids = None

@app.get("/")
async def get():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

async def process_stream(websocket: WebSocket, mode: str):
    await websocket.accept()
    print(f"Client connected to {mode} mode")
    
    # Create a new streamer instance for this connection, sharing the model
    streamer = WhisperStreamer(model=model, processor=processor, device=DEVICE)
    
    # "standard" = use_cache=False (slower decoding)
    # "optimized" = use_cache=True (faster decoding)
    use_cache = (mode == "optimized")
    
    async def receive_audio():
        try:
            while True:
                data = await websocket.receive_bytes()
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                streamer.push_audio(audio_chunk)
        except Exception:
            pass # Disconnect handled by main wait

    async def send_transcription():
        try:
            while True:
                # Transcribe every 0.5s
                await asyncio.sleep(0.5)
                
                start_time = time.time()
                # Run blocking inference in a separate thread
                text = await asyncio.to_thread(streamer.transcribe_step, use_cache=use_cache)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                
                response = {
                    "text": text,
                    "latency": latency_ms
                }
                await websocket.send_text(json.dumps(response))
        except Exception as e:
            print(f"Transcription error in {mode}: {e}")

    try:
        # Run both loops concurrently
        receiver = asyncio.create_task(receive_audio())
        sender = asyncio.create_task(send_transcription())
        
        # Wait until either fails (likely receiver disconnects)
        done, pending = await asyncio.wait(
            [receiver, sender],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            
    except WebSocketDisconnect:
        print(f"Client disconnected from {mode}")
    except Exception as e:
        print(f"Error in {mode}: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/ws/{mode}")
async def websocket_endpoint(websocket: WebSocket, mode: str):
    await process_stream(websocket, mode)

@app.websocket("/ws")
async def websocket_default(websocket: WebSocket):
    await process_stream(websocket, "optimized")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
