import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .audio_utils import AudioBuffer
import numpy as np

class WhisperStreamer:
    def __init__(self, model_id="openai/whisper-tiny.en", device="cpu", model=None, processor=None):
        self.device = device
        
        if model and processor:
            self.model = model
            self.processor = processor
        else:
            print(f"Loading model {model_id} on {device}...")
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
            self.model.config.forced_decoder_ids = None
        
        # Sliding window buffer (30 seconds max context)
        self.buffer = AudioBuffer(sample_rate=16000, window_size_seconds=30)
        
    def push_audio(self, chunk: np.ndarray):
        """Push raw audio chunk (float32, 16kHz) to buffer."""
        self.buffer.push(chunk)

    def transcribe_step(self, use_cache=True):
        """
        Process the current audio buffer and return the transcription.
        """
        audio_window = self.buffer.get_window()
        
        # Preprocessing (Mel Spectrogram)
        input_features = self.processor(
            audio_window, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)

        # Generation
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                use_cache=use_cache, 
                max_new_tokens=64, 
            )

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
