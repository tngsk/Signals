"""
Context-aware DSP utilities that wrap core DSP functions.

This module provides context-aware wrappers for the core DSP functions,
allowing them to automatically use the current synthesis context's sample rate
while maintaining backward compatibility with explicit parameter specification.
"""

from typing import Optional
import numpy as np

from .dsp import generate_silence as _generate_silence, write_wav as _write_wav
from .context import SynthContext, ContextError


def generate_silence(silence_duration: float, sample_rate: Optional[int] = None) -> np.ndarray:
    """
    Generate silence with automatic sample rate from context.
    
    Context-aware wrapper for core.dsp.generate_silence that automatically
    uses the current synthesis context's sample rate if available.
    
    Args:
        silence_duration: Duration of silence in seconds (must be >= 0.0)
        sample_rate: Sample rate in Hz. If None, uses current context sample rate
        
    Returns:
        Numpy array of zeros with shape (num_samples,) and dtype float32
        
    Raises:
        ContextError: If no sample_rate provided and no context available
        
    Example:
        >>> with synthesis_context(48000):
        ...     silence = generate_silence(1.0)  # Uses 48kHz from context
        ...     print(silence.shape)
        (48000,)
        
        >>> # Or with explicit sample rate (overrides context)
        >>> silence = generate_silence(1.0, sample_rate=44100)
    """
    if sample_rate is None:
        try:
            sample_rate = SynthContext.get_sample_rate()
        except ContextError:
            raise ContextError(
                "No sample_rate provided and no synthesis context available. "
                "Either provide sample_rate explicitly or use within a synthesis context."
            )
    
    return _generate_silence(silence_duration, sample_rate)


def write_wav(
    filename: str, 
    samples: np.ndarray, 
    sample_rate: Optional[int] = None, 
    bits_per_sample: int = 16
):
    """
    Write audio samples to WAV file with automatic sample rate from context.
    
    Context-aware wrapper for core.dsp.write_wav that automatically
    uses the current synthesis context's sample rate if available.
    
    Args:
        filename: Output WAV file path
        samples: Audio samples as numpy array with values in range [-1.0, 1.0]
        sample_rate: Sample rate in Hz. If None, uses current context sample rate
        bits_per_sample: Bit depth for output (16, 24, or 32 bits)
        
    Raises:
        ContextError: If no sample_rate provided and no context available
        ValueError: If bits_per_sample is not 16, 24, or 32
        
    Example:
        >>> with synthesis_context(48000):
        ...     samples = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        ...     write_wav("output.wav", samples)  # Uses 48kHz from context
        
        >>> # Or with explicit sample rate (overrides context)
        >>> write_wav("output.wav", samples, sample_rate=44100)
    """
    if sample_rate is None:
        try:
            sample_rate = SynthContext.get_sample_rate()
        except ContextError:
            raise ContextError(
                "No sample_rate provided and no synthesis context available. "
                "Either provide sample_rate explicitly or use within a synthesis context."
            )
    
    return _write_wav(filename, samples, sample_rate, bits_per_sample)


def get_context_sample_rate() -> int:
    """
    Get the current context's sample rate.
    
    Utility function to access the current synthesis context's sample rate.
    Useful for applications that need to query the sample rate without
    creating audio objects.
    
    Returns:
        Current context sample rate in Hz
        
    Raises:
        ContextError: If no synthesis context is active
        
    Example:
        >>> with synthesis_context(48000):
        ...     rate = get_context_sample_rate()
        ...     print(f"Current sample rate: {rate}Hz")
        Current sample rate: 48000Hz
    """
    return SynthContext.get_sample_rate()


def has_context() -> bool:
    """
    Check if a synthesis context is currently active.
    
    Returns:
        True if a synthesis context is active, False otherwise
        
    Example:
        >>> print(has_context())  # Outside context
        False
        >>> with synthesis_context(48000):
        ...     print(has_context())  # Inside context
        True
    """
    return SynthContext.has_context()


# Re-export core DSP functions for direct access when needed
from .dsp import generate_silence as generate_silence_explicit
from .dsp import write_wav as write_wav_explicit

__all__ = [
    'generate_silence',
    'write_wav', 
    'get_context_sample_rate',
    'has_context',
    'generate_silence_explicit',
    'write_wav_explicit'
]