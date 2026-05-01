"""
Core components for the Signals synthesizer framework.

This module contains the fundamental building blocks used throughout the system:
- Module: Base class for all signal processing modules
- Signal: Container for typed signal data
- DSP utilities: Low-level audio processing functions
"""

from .context import (
    ContextError,
    SynthContext,
    get_sample_rate_or_default,
    require_context,
    synthesis_context,
)
from .dsp import generate_silence, write_wav
from .logging import (
    LogContext,
    LogLevel,
    configure_logging,
    enable_performance_logging,
    get_logger,
    log_module_state,
    log_signal_info,
    performance_logger,
    set_module_log_level,
)
from .module import Module, ParameterType, Signal, SignalType

__all__ = [
    "Module",
    "Signal",
    "SignalType",
    "ParameterType",
    "write_wav",
    "generate_silence",
    # Logging
    "get_logger",
    "configure_logging",
    "set_module_log_level",
    "enable_performance_logging",
    "performance_logger",
    "LogContext",
    "LogLevel",
    "log_module_state",
    "log_signal_info",
    # Context management
    "SynthContext",
    "synthesis_context",
    "get_sample_rate_or_default",
    "require_context",
    "ContextError"
]
