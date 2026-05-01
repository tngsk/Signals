"""
Digital Signal Processing utilities for audio file operations.

This module provides low-level DSP functions for audio file I/O and basic
audio processing operations. It includes functions for generating silence
and writing audio data to WAV files with support for different bit depths.
"""

import struct
import wave

import numpy as np


def generate_silence(silence_duration: float, sample_rate: int) -> np.ndarray:
    """
    Generate a numpy array containing silence of specified duration.

    Creates an array of zeros representing digital silence for the given
    duration and sample rate. This is useful for adding pauses, lead-ins,
    or spacing between audio segments.

    Args:
        silence_duration: Duration of silence in seconds (must be >= 0.0)
        sample_rate: Audio sample rate in Hz (must be > 0)

    Returns:
        Numpy array of zeros with shape (num_samples,) and dtype float32

    Example:
        >>> silence = generate_silence(1.0, 48000)  # 1 second at 48kHz
        >>> print(silence.shape)
        (48000,)
        >>> print(np.all(silence == 0.0))
        True
    """
    num_samples = int(silence_duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def write_wav(
    filename: str, samples: np.ndarray, sample_rate: int, bits_per_sample: int = 16
):
    """
    Write a numpy array of audio samples to a WAV file.

    Converts floating-point audio samples in the range [-1.0, 1.0] to the
    specified bit depth and writes them to a standard WAV file. The function
    handles proper scaling and format conversion for different bit depths.

    Args:
        filename: Output WAV file path
        samples: Audio samples as numpy array with values in range [-1.0, 1.0]
        sample_rate: Audio sample rate in Hz
        bits_per_sample: Bit depth for output (16, 24, or 32 bits)

    Raises:
        ValueError: If bits_per_sample is not 16, 24, or 32

    Note:
        - 16-bit output uses signed integer encoding
        - 24-bit and 32-bit float formats may have limited player compatibility
        - Samples outside [-1.0, 1.0] range may cause clipping

    Example:
        >>> samples = np.array([0.5, -0.3, 0.8, -0.1], dtype=np.float32)
        >>> write_wav("output.wav", samples, 48000, 16)
    """
    if bits_per_sample not in [16, 24, 32]:
        raise ValueError("bits_per_sample must be 16, 24, or 32")

    num_channels = 1  # Mono for now
    sample_width = bits_per_sample // 8

    with wave.open(filename, "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)

        if bits_per_sample == 16:
            scaled_samples = (samples * 32767).astype(np.int16)
            for sample in scaled_samples:
                wf.writeframesraw(struct.pack("<h", sample))
        elif bits_per_sample == 24:
            scaled_samples = (samples * 8388607).astype(np.int32)
            for sample in scaled_samples:
                # Pack as 3 bytes (24-bit) in little-endian format
                sample_bytes = sample.to_bytes(4, byteorder='little', signed=True)[:3]
                wf.writeframesraw(sample_bytes)
        else:  # 32-bit float
            for sample in samples.astype(np.float32):
                wf.writeframesraw(struct.pack("<f", sample))
