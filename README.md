# Memory-Efficient Whisper Streaming

This project implements a real-time, memory-efficient streaming architecture for OpenAI's Whisper model. It uses a **Sliding Window** approach with **KV Caching** (via Hugging Face Transformers) to enable continuous transcription without exploding memory usage.

## Architecture

1.  **Chunked Streaming**: Audio is processed in small chunks (e.g., 200ms) rather than waiting for the full file.
2.  **Sliding Window**: We maintain a fixed-size buffer (e.g., 30 seconds) of the most recent audio. This ensures the Encoder input never grows beyond O(N).
3.  **KV Cache Reuse**: The Decoder uses Key-Value caching (standard in `transformers`) to speed up the autoregressive text generation for each window.

## Project Structure

- `whisper_streamer/`: Core library.
    - `core.py`: `WhisperStreamer` class managing the model and buffer.
    - `audio_utils.py`: Rolling audio buffer implementation.
- `server.py`: FastAPI WebSocket server that handles audio streams.
- `client.py`: Python client that streams microphone audio to the server.
- `benchmark.ipynb`: Notebook to demonstrate and benchmark the streaming logic.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You need `ffmpeg` installed on your system for Whisper.*

## Usage

### 1. Run the Server
Start the WebSocket server:
```bash
python server.py
```

### 2. Run the Client
In a separate terminal, start the microphone client:
```bash
python client.py
```
Speak into your microphone, and you should see the transcript updating in real-time!

### 3. Run the Benchmark
Open `benchmark.ipynb` in VS Code or Jupyter to see the internal logic applied to synthetic audio.

## Why this matters?
Standard Whisper processes the *entire* audio context. If you stream for 1 hour:
- **Standard**: Attention complexity grows to O(T^2). Memory explodes. Latency increases.
- **This Project**: Attention stays O(Window^2). Memory is constant. Latency is stable.
