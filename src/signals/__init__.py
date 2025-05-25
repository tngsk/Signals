"""
Signals package for audio synthesis and signal processing.

This package provides a modular synthesizer framework with the following components:

- Core: Base classes and utilities (Module, Signal, DSP functions)
- Modules: Signal processing modules (Oscillator, Envelope, Mixer, Output)
- Processing: High-level synthesis engines (SynthEngine, ModuleGraph, Patch)

Example:
    Basic synthesis setup:

    >>> from signals import SynthEngine
    >>> engine = SynthEngine(sample_rate=48000)
    >>> patch = engine.load_patch("synth.yaml")
    >>> audio = engine.render(duration=2.0)
"""

# Import from reorganized structure
from .core import Module, Signal, SignalType, ParameterType
from .core import get_logger, configure_logging, set_module_log_level, LogLevel
from .core import SynthContext, synthesis_context, get_sample_rate_or_default, ContextError
from .core.dsp_context import (
    generate_silence, write_wav,
    get_context_sample_rate, has_context,
    generate_silence_explicit, write_wav_explicit
)
from .modules import Oscillator, EnvelopeADSR, Mixer, OutputWav, VCA, Filter
from .modules.oscillator import WaveformType
from .modules.filter import FilterType
from .processing import SynthEngine, ModuleGraph, Patch, PatchTemplate

# Maintain backward compatibility with old imports
__all__ = [
    # Core components
    "Module",
    "Signal", 
    "SignalType",
    "ParameterType",
    
    # Context management
    "SynthContext",
    "synthesis_context",
    "get_sample_rate_or_default",
    "ContextError",
    
    # Context-aware DSP functions (primary API)
    "generate_silence",
    "write_wav",
    "get_context_sample_rate",
    "has_context",
    
    # Explicit DSP functions (for when context override needed)
    "generate_silence_explicit",
    "write_wav_explicit",
    
    # Logging
    "get_logger",
    "configure_logging",
    "set_module_log_level",
    "LogLevel",
    
    # Modules
    "Oscillator",
    "WaveformType",
    "EnvelopeADSR",
    "Mixer", 
    "OutputWav",
    "VCA",
    "Filter",
    "FilterType",
    
    # Processing engines
    "SynthEngine",
    "ModuleGraph", 
    "Patch",
    "PatchTemplate"
]