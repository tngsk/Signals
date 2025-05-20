from .module import Module, Signal, SignalType, ParameterType
from .dsp import write_wav, generate_silence
import numpy as np

class OutputWav(Module):
    def __init__(self, filename: str, sample_rate: int, bits_per_sample: int = 16):
        super().__init__(input_count=1, output_count=0)
        self.filename = filename
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self._buffer = []

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        if inputs and inputs[0].type == SignalType.AUDIO:
            # For now, assume single float values. Block processing later.
            self._buffer.append(inputs[0].value)
        return []
        
    def add_silence_at_start(self, silence_duration: float):
        """
        Adds silence of specified duration in seconds at the start of the audio buffer.
        
        Args:
            silence_duration: Duration of silence in seconds
        """
        if not self._buffer:
            print("Warning: Buffer is empty, adding silence to an empty buffer")
            
        # Generate silence
        silence = generate_silence(silence_duration, self.sample_rate)
        
        # Convert current buffer to numpy array
        current_buffer = np.array(self._buffer, dtype=np.float32)
        
        # Concatenate silence with current buffer
        new_buffer = np.concatenate((silence, current_buffer))
        
        # Update buffer
        self._buffer = new_buffer.tolist()

    def finalize(self):
        if self._buffer:
            samples_np = np.array(self._buffer, dtype=np.float32)
            write_wav(self.filename, samples_np, self.sample_rate, self.bits_per_sample)
            print(f"Audio written to {self.filename}")
        self._buffer = [] # Clear buffer
