# Memory-Efficient Whisper Streaming: Real-Time ASR Optimization 

| Status | Model | Framework | Architecture |
| :--- | :--- | :--- | :--- |
| Active | OpenAI Whisper | Hugging Face Transformers, PyTorch | Sliding Window, KV Caching |

##  Overview

This project delivers a robust, real-time Speech-to-Text (ASR) streaming solution built upon OpenAI's Whisper model. It successfully tackles the primary challenges of standard ASR models in a continuous streaming environment: unbounded memory growth and increasing inference latency. By integrating a Sliding Window Attention mechanism with native Key-Value (KV) Caching, we ensure both constant memory footprint and stable, low-latency transcription performance, making it production-ready for long-duration applications.

## System Architecture

The system employs a Client-Server architecture utilizing WebSockets for high-speed, persistent, bi-directional communication.

*   **Client (Microphone):** Captures raw 16kHz audio, chunks it (e.g., 200ms), and streams it to the server.
*   **FastAPI Server:** Orchestrates the process, managing the WebSocket connection and decoupling network I/O from the compute-intensive model inference using `asyncio.to_thread()`.
*   **Whisper Streamer Core:** Manages the audio buffer and applies the optimizations before calling the Hugging Face model.
*   **Optimized Inference:** The model runs on the latest 30-second audio window, reusing past decoder states via KV Caching.

## Technical Deep Dive: The Optimizations

### 1. Sliding Window Attention (Fixed Buffer)

The Whisper Encoder input is limited to 30 seconds of audio. In streaming, a naive approach would continually append audio, leading to a Context Window Overflow and $O(N)$ memory growth over time.

*   **Implementation:** The `audio_utils.py` implements a fixed-size rolling buffer (30 seconds $\approx 480,000$ samples). When a new chunk arrives, the oldest chunk of the same size is discarded, ensuring the buffer's memory footprint remains strictly $O(1)$ with respect to total stream time.
*   **Benefit:** Guarantees stable memory usage and a fixed $O(\text{Window}^2)$ complexity for the Encoder's self-attention, regardless of how long the user speaks.

### 2. Key-Value (KV) Caching for Decoder

The Whisper Decoder generates text tokens autoregressively. Without caching, the model re-computes the attention keys ($K$) and values ($V$) for all previously generated tokens at every new decoding step.

*   **Implementation:** By setting `use_cache=True` in the Hugging Face `model.generate()` call, we instruct the model to store $K$ and $V$ tensors from previous steps. In subsequent decoding steps, only the $K$ and $V$ for the new token are computed; the rest are retrieved from the cache.
*   **Benefit:** This optimization reduces the Decoder's complexity from quadratic $\sum_{i=1}^{L} i \approx O(L^2)$ to linear $O(L)$ for a sequence of length $L$. This drastically reduces inference time, preventing Latency Drift.

## Performance Advantage

| Feature | Standard (Naive) Approach | Optimized Streaming Approach |
| :--- | :--- | :--- |
| Memory | $O(T)$ (grows indefinitely with time $T$) | $O(1)$ (Constant) |
| Encoder Complexity | $O(T^2)$ | $O(\text{Window}^2)$ (Constant) |
| Decoder Complexity | $O(L^2)$ | $O(L)$ (Linear) |
| Latency | Increases linearly (Latency Drift) | Stable and Low |

The results in `benchmark.ipynb` consistently show a 30-50% reduction in end-to-end latency compared to the uncached approach, bringing the transcription closer to real-time.

## Project Structure

```text
├── whisper_streamer/
│   ├── core.py               # `WhisperStreamer` class managing the model, cache, and inference loop.
│   └── audio_utils.py        # Optimized rolling audio buffer implementation (Sliding Window).
├── server.py                 # FastAPI WebSocket server handling concurrent connections.
├── client.py                 # Pyaudio-based client for microphone streaming.
├── benchmark.ipynb           # Comparative notebook for latency and memory analysis.
└── requirements.txt
```

## Setup

### Prerequisites

You must have **FFmpeg** installed on your system.

*   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
*   **macOS (Homebrew):** `brew install ffmpeg`

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/mounibnasr45/Memory-Efficient-with-OpenAI-Whisper
    cd memory-efficient-whisper-streaming
    ```

2.  Install required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: This includes `pyaudio` for the client, which might require system dependencies on some Linux distributions.)*

## Usage

### 1. Run the Server

Start the WebSocket server on `localhost:8000`:

```bash
python server.py
```

### 2. Run the Client

In a separate terminal, start the microphone client. It will connect to the server and begin streaming:

```bash
python client.py
```

Speak into your microphone. The server terminal will output the transcription with measured latency for each chunk.
