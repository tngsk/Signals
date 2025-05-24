"""
Core components for the Signals synthesizer framework.

This module contains the fundamental building blocks used throughout the system:
- Module: Base class for all signal processing modules
- Signal: Container for typed signal data
- DSP utilities: Low-level audio processing functions
"""

from .module import Module, Signal, SignalType, ParameterType
from .dsp import write_wav, generate_silence
from .logging import (
    get_logger, configure_logging, set_module_log_level,
    enable_performance_logging, performance_logger,
    LogContext, LogLevel, log_module_state, log_signal_info
)

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
    "log_signal_info"
]