import numpy as np

class AudioBuffer:
    def __init__(self, sample_rate=16000, window_size_seconds=30):
        self.sample_rate = sample_rate
        self.window_size_samples = int(sample_rate * window_size_seconds)
        self.buffer = np.zeros(self.window_size_samples, dtype=np.float32)
        self.new_samples_count = 0

    def push(self, chunk: np.ndarray):
        """
        Push a new chunk of audio into the rolling buffer.
        The buffer slides to the left to accommodate the new chunk.
        """
        chunk_len = len(chunk)
        if chunk_len > self.window_size_samples:
            # If chunk is larger than window, just take the last window_size
            self.buffer[:] = chunk[-self.window_size_samples:]
        else:
            # Shift buffer to left
            self.buffer = np.roll(self.buffer, -chunk_len)
            # Overwrite end with new chunk
            self.buffer[-chunk_len:] = chunk
        
        self.new_samples_count += chunk_len

    def get_window(self):
        """Return the current full window of audio."""
        return self.buffer

    def clear(self):
        self.buffer.fill(0)
        self.new_samples_count = 0
