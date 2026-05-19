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
from .core import (
    ContextError,
    LogLevel,
    Module,
    ParameterType,
    Signal,
    SignalType,
    SynthContext,
    configure_logging,
    get_logger,
    get_sample_rate_or_default,
    set_module_log_level,
    synthesis_context,
)
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

    # Logging
    "get_logger",
    "configure_logging",
    "set_module_log_level",
    "LogLevel",
]
