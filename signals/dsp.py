import wave
import struct
import numpy as np

def generate_silence(silence_duration: float, sample_rate: int) -> np.ndarray:
    """
    Generates a numpy array containing silence of specified duration in seconds.
    
    Args:
        silence_duration: Duration of silence in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Numpy array containing zeros (silence)
    """
    num_samples = int(silence_duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def write_wav(filename: str, samples: np.ndarray, sample_rate: int, bits_per_sample: int = 16):
    """
    Writes a numpy array of samples to a WAV file.
    Assumes samples are in the range [-1.0, 1.0].
    """
    if bits_per_sample not in [16, 24, 32]:
        raise ValueError("bits_per_sample must be 16, 24, or 32")

    num_channels = 1 # Mono for now
    sample_width = bits_per_sample // 8

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)

        if bits_per_sample == 16:
            scaled_samples = (samples * 32767).astype(np.int16)
            for sample in scaled_samples:
                wf.writeframesraw(struct.pack('<h', sample))
        # Add 24-bit and 32-bit float support later if needed
        else: # For 32-bit float (though wave module might not fully support it directly for all players)
            for sample in samples.astype(np.float32): # wave module expects bytes
                 wf.writeframesraw(sample.tobytes())
