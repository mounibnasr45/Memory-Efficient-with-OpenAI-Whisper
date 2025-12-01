# Memory-Efficient Whisper Streaming Project

## 1. Project Overview
This project implements a real-time speech-to-text streaming system using OpenAI's Whisper model. The core innovation lies in its **memory-efficient architecture**, which enables continuous transcription of long audio streams without running out of memory or suffering from increasing latency.

It achieves this through two main techniques:
1.  **Sliding Window Attention**: Keeps a fixed-size context (e.g., 30 seconds) to prevent memory growth.
2.  **KV Caching**: Reuses previously computed key-value pairs in the self-attention mechanism to speed up token generation.

## 2. System Architecture

The system follows a **Client-Server** architecture using WebSockets for low-latency communication.

```mermaid
graph LR
    A[Client (Mic/Browser)] -- Raw Audio (Float32) --> B[FastAPI Server]
    B -- Push Audio --> C[Audio Buffer]
    C -- Get Window (30s) --> D[Whisper Encoder]
    D -- Features --> E[Whisper Decoder]
    E -- Text --> B
    B -- JSON (Text + Latency) --> A
```

### Components
1.  **Client**: Captures audio from the microphone (Python script or Web Browser).
2.  **Server**: A FastAPI application that manages WebSocket connections.
3.  **Streamer Core**: Handles the audio buffering and model inference.

## 3. Key Interventions & Optimizations

### A. Sliding Window Buffer (`audio_utils.py`)
Standard Whisper processes 30-second chunks. If we simply appended new audio forever, the buffer would grow indefinitely.
*   **Our Approach**: We use a "Rolling Buffer" (implemented via `numpy.roll`).
*   **How it works**: When new audio arrives, the old audio is shifted out. The buffer size remains constant (e.g., exactly 30 seconds).

### B. KV Caching (`core.py`)
In Transformer models, generating the next token usually requires re-calculating attention for all previous tokens.
*   **Standard Approach**: $O(N^2)$ complexity for generating $N$ tokens.
*   **Optimized Approach (KV Cache)**: We cache the Key and Value matrices of previous tokens. The model only computes attention for the *new* token.
*   **Impact**: drastically reduces latency during the text generation phase.

## 4. Code Breakdown

### 1. `whisper_streamer/audio_utils.py`
**Role**: Manages the raw audio data.
*   **`AudioBuffer` Class**:
    *   Maintains a fixed-size `numpy` array (e.g., $16000 \times 30$ samples).
    *   `push(chunk)`: Rolls the array and inserts new data at the end.

### 2. `whisper_streamer/core.py`
**Role**: Wraps the Hugging Face Whisper model.
*   **`transcribe_step(use_cache=True)`**:
    *   Gets the last 30s of audio from the buffer.
    *   Converts audio to Mel Spectrogram features.
    *   Calls `model.generate()` with `use_cache=True` to enable the optimization.

### 3. `server.py`
**Role**: The backend engine.
*   **Async Concurrency**: Uses `asyncio` to handle network I/O (receiving audio) separately from CPU-bound tasks (inference).
*   **Threading**: Runs the heavy `transcribe_step` in a separate thread (`asyncio.to_thread`) so it doesn't block the WebSocket heartbeat.
*   **Endpoints**:
    *   `/ws/standard`: Runs without KV cache (for comparison).
    *   `/ws/optimized`: Runs with KV cache.

### 4. `client.py`
**Role**: A Python CLI client.
*   Uses `sounddevice` to capture raw audio from the microphone.
*   Sends audio chunks to the server via WebSockets.
*   Receives and prints the transcription with latency metrics.

### 5. `templates/index.html`
**Role**: Visualization Dashboard.
*   Uses the Web Audio API to capture microphone input in the browser.
*   Connects to both "Standard" and "Optimized" endpoints simultaneously.
*   Renders a real-time chart comparing the latency of both approaches.

## 5. Workflow Example

1.  **Capture**: The user says "Hello World". The client captures 0.25s of audio (4000 samples).
2.  **Transmission**: The client sends these bytes to `ws://localhost:8000/ws`.
3.  **Buffering**:
    *   Server receives the bytes.
    *   `AudioBuffer` shifts the previous 29.75s of audio to the left.
    *   The new 0.25s is appended to the right.
4.  **Inference Trigger**: Every 0.5 seconds, the server triggers a transcription.
5.  **Processing**:
    *   The full 30s buffer is converted to a spectrogram.
    *   The Whisper Decoder generates tokens ("Hello", " World").
    *   *Optimization*: It uses cached attention weights to generate these tokens faster.
6.  **Response**: The server sends `{"text": "Hello World", "latency": 450}` back to the client.
7.  **Display**: The client updates the screen.
