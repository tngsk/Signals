"""
Output module for writing audio signals to WAV files.

This module provides the OutputWav class which captures audio signals from the
synthesizer pipeline and writes them to WAV audio files. It supports various
bit depths and includes functionality for adding silence to recordings.
"""

from .module import Module, Signal, SignalType, ParameterType
from .dsp import write_wav, generate_silence
import numpy as np

class OutputWav(Module):
    """
    WAV file output module for capturing audio signals.
    
    The OutputWav module acts as a sink in the synthesizer pipeline, capturing
    incoming audio signals and buffering them for eventual output to a WAV file.
    It supports different bit depths and provides utilities for audio manipulation
    such as adding silence.
    
    Args:
        filename: Name of the output WAV file
        sample_rate: Audio sample rate in Hz
        bits_per_sample: Bit depth for audio encoding (default: 16)
        
    Attributes:
        filename (str): Output file name
        sample_rate (int): Sample rate for audio output
        bits_per_sample (int): Bit depth (16, 24, or 32 bits)
        
    Example:
        >>> output = OutputWav("recording.wav", sample_rate=48000, bits_per_sample=24)
        >>> output.process([audio_signal])  # Capture audio
        >>> output.add_silence_at_start(0.5)  # Add 0.5s silence
        >>> output.finalize()  # Write to file
    """
    def __init__(self, filename: str, sample_rate: int, bits_per_sample: int = 16):
        super().__init__(input_count=1, output_count=0)
        self.filename = filename
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self._buffer = []

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Capture incoming audio signals into the internal buffer.
        
        Processes one audio sample at a time and stores it in the internal buffer
        for later output to the WAV file. Only AUDIO type signals are captured;
        other signal types are ignored.
        
        Args:
            inputs: List of input signals. Only the first AUDIO signal is captured.
            
        Returns:
            Empty list (output modules produce no signals)
            
        Note:
            Currently supports single-sample processing. Block processing
            will be added in future versions for improved efficiency.
        """
        if inputs and inputs[0].type == SignalType.AUDIO:
            # For now, assume single float values. Block processing later.
            self._buffer.append(inputs[0].value)
        return []
        
    def add_silence_at_start(self, silence_duration: float):
        """
        Add silence of specified duration at the beginning of the audio buffer.
        
        Prepends a period of silence (zero values) to the beginning of the captured
        audio. This is useful for creating lead-in time or spacing between audio
        segments. The silence is inserted before any existing audio data.
        
        Args:
            silence_duration: Duration of silence to add in seconds (must be >= 0.0)
            
        Warning:
            If the buffer is empty when this method is called, it will still add
            silence, which may not be the intended behavior.
            
        Example:
            >>> output.add_silence_at_start(1.0)  # Add 1 second of silence
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
        """
        Write the captured audio buffer to a WAV file and clear the buffer.
        
        Converts the internal audio buffer to the specified bit depth and writes
        it to the output WAV file. After writing, the internal buffer is cleared
        to free memory and prepare for potential reuse of the module.
        
        Note:
            This method must be called to actually create the output file.
            If the buffer is empty, no file will be written.
            
        Example:
            >>> output.finalize()  # Creates the WAV file
            Audio written to output.wav
        """
        if self._buffer:
            samples_np = np.array(self._buffer, dtype=np.float32)
            write_wav(self.filename, samples_np, self.sample_rate, self.bits_per_sample)
            print(f"Audio written to {self.filename}")
        self._buffer = [] # Clear buffer
