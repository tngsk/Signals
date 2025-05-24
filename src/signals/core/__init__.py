"""
Core components for the Signals synthesizer framework.

This module contains the fundamental building blocks used throughout the system:
- Module: Base class for all signal processing modules
- Signal: Container for typed signal data
- DSP utilities: Low-level audio processing functions
"""

from .module import Module, Signal, SignalType, ParameterType
from .dsp import write_wav, generate_silence

__all__ = [
    "Module",
    "Signal", 
    "SignalType",
    "ParameterType",
    "write_wav",
    "generate_silence"
]