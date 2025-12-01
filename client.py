import asyncio
import websockets
import sounddevice as sd
import numpy as np
import sys
import json

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 4000  # 0.25 seconds per chunk

async def microphone_client():
    uri = "ws://localhost:8000/ws"
    
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as websocket:
        print("Connected! Speak into your microphone...")
        
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            # Copy data to avoid issues with buffer reuse
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

        # Start microphone stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=BLOCKSIZE,
            callback=callback
        )
        
        with stream:
            # Create separate tasks for sending and receiving
            async def send_audio():
                while True:
                    data = await audio_queue.get()
                    await websocket.send(data.tobytes())

            async def receive_text():
                try:
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            text = data.get("text", "")
                            latency = data.get("latency", 0)
                            
                            # Prepare output string
                            output = f"[{latency:.0f}ms] Transcript: {text}"
                            
                            # Print with padding to clear previous characters (simulating clear line)
                            # Using a fixed width of 120 characters or more to ensure overwrite
                            print(f"\r{output.ljust(150)}", end="", flush=True)
                        except json.JSONDecodeError:
                            print(f"\rTranscript: {message}", end="", flush=True)
                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection closed by server.")

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_text())

            # Wait until either task finishes (likely recv_task on connection close)
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the other task
            for task in pending:
                task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(microphone_client())
    except KeyboardInterrupt:
        print("\nStopped.")
