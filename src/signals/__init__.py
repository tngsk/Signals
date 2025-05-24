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
from .core import Module, Signal, SignalType, ParameterType, write_wav
from .modules import Oscillator, EnvelopeADSR, Mixer, OutputWav
from .processing import SynthEngine, ModuleGraph, Patch, PatchTemplate

# Maintain backward compatibility with old imports
__all__ = [
    # Core components
    "Module",
    "Signal", 
    "SignalType",
    "ParameterType",
    "write_wav",
    
    # Modules
    "Oscillator",
    "EnvelopeADSR",
    "Mixer", 
    "OutputWav",
    
    # Processing engines
    "SynthEngine",
    "ModuleGraph", 
    "Patch",
    "PatchTemplate"
]