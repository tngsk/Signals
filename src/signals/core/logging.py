"""
Comprehensive logging system for the Signals synthesizer framework.

This module provides a centralized logging system with hierarchical loggers,
configurable output levels, and performance-optimized logging for audio processing.
Supports both console and file output with structured formatting.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum
import time
import functools


class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class SignalsLogger:
    """
    Centralized logger for the Signals framework.
    
    Provides hierarchical logging with module-specific loggers, configurable
    output destinations, and performance-optimized logging for real-time audio processing.
    """
    
    _instance: Optional['SignalsLogger'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'SignalsLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._loggers: Dict[str, logging.Logger] = {}
        self._log_level = LogLevel.INFO
        self._console_enabled = True
        self._file_enabled = False
        self._file_path: Optional[Path] = None
        self._formatter = self._create_formatter()
        self._performance_logging = False
        
        # Initialize root logger
        self._setup_root_logger()
        self._initialized = True
    
    def _create_formatter(self) -> logging.Formatter:
        """Create a structured log formatter."""
        return logging.Formatter(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _setup_root_logger(self):
        """Setup the root logger for the Signals framework."""
        root_logger = logging.getLogger('signals')
        root_logger.setLevel(self._log_level.value)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if self._console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self._file_enabled and self._file_path:
            file_handler = logging.handlers.RotatingFileHandler(
                self._file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(self._formatter)
            root_logger.addHandler(file_handler)
        
        self._loggers['signals'] = root_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module.
        
        Args:
            name: Logger name (e.g., 'signals.modules.oscillator')
            
        Returns:
            Logger instance for the specified module
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self._log_level.value)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def configure(self, 
                  level: Union[LogLevel, str] = LogLevel.INFO,
                  console: bool = True,
                  file_path: Optional[Union[str, Path]] = None,
                  performance_logging: bool = False):
        """
        Configure the logging system.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console: Enable console output
            file_path: Optional file path for log output
            performance_logging: Enable performance logging for audio processing
        """
        # Convert string level to enum
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        
        self._log_level = level
        self._console_enabled = console
        self._file_enabled = file_path is not None
        self._file_path = Path(file_path) if file_path else None
        self._performance_logging = performance_logging
        
        # Recreate root logger with new settings
        self._setup_root_logger()
        
        # Update existing loggers
        for logger in self._loggers.values():
            logger.setLevel(level.value)
    
    def set_module_level(self, module_name: str, level: Union[LogLevel, str]):
        """
        Set log level for a specific module.
        
        Args:
            module_name: Module name (e.g., 'oscillator', 'envelope')
            level: Log level for this module
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        
        full_name = f"signals.modules.{module_name}"
        logger = self.get_logger(full_name)
        logger.setLevel(level.value)
    
    def enable_performance_logging(self, enabled: bool = True):
        """Enable or disable performance logging."""
        self._performance_logging = enabled
    
    def is_performance_logging_enabled(self) -> bool:
        """Check if performance logging is enabled."""
        return self._performance_logging


# Global logger instance
_signals_logger = SignalsLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: Module name (will be prefixed with 'signals.')
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger('modules.oscillator')
        >>> logger.info("Oscillator initialized")
    """
    if not name.startswith('signals.'):
        name = f"signals.{name}"
    return _signals_logger.get_logger(name)


def configure_logging(**kwargs):
    """
    Configure the global logging system.
    
    Args:
        **kwargs: Configuration options (see SignalsLogger.configure)
        
    Example:
        >>> configure_logging(level='DEBUG', file_path='signals.log')
    """
    _signals_logger.configure(**kwargs)


def set_module_log_level(module_name: str, level: Union[LogLevel, str]):
    """
    Set log level for a specific module.
    
    Args:
        module_name: Module name
        level: Log level
        
    Example:
        >>> set_module_log_level('oscillator', 'DEBUG')
    """
    _signals_logger.set_module_level(module_name, level)


def enable_performance_logging(enabled: bool = True):
    """Enable or disable performance logging."""
    _signals_logger.enable_performance_logging(enabled)


def performance_logger(func):
    """
    Decorator for performance-critical functions.
    
    Logs execution time when performance logging is enabled.
    Minimal overhead when disabled.
    
    Example:
        @performance_logger
        def process_audio(self, inputs):
            # ... processing code
            return outputs
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _signals_logger.is_performance_logging_enabled():
            return func(*args, **kwargs)
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Get logger for the function's module
        module_name = func.__module__.replace('signals.', '') if func.__module__ else 'unknown'
        logger = get_logger(module_name)
        
        logger.debug(f"{func.__qualname__}: {execution_time:.3f}ms")
        
        return result
    
    return wrapper


class LogContext:
    """
    Context manager for temporary log level changes.
    
    Example:
        with LogContext('modules.oscillator', 'DEBUG'):
            # Oscillator logs will be at DEBUG level
            oscillator.process()
        # Log level restored
    """
    
    def __init__(self, logger_name: str, level: Union[LogLevel, str]):
        self.logger_name = logger_name
        self.new_level = LogLevel[level.upper()] if isinstance(level, str) else level
        self.original_level = None
        self.logger = None
    
    def __enter__(self):
        self.logger = get_logger(self.logger_name)
        self.original_level = self.logger.level
        self.logger.setLevel(self.new_level.value)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger and self.original_level is not None:
            self.logger.setLevel(self.original_level)


def log_module_state(logger: logging.Logger, module_name: str, state: Dict[str, Any]):
    """
    Log module state information in a structured format.
    
    Args:
        logger: Logger instance
        module_name: Name of the module
        state: Dictionary of state information
    """
    if logger.isEnabledFor(logging.DEBUG):
        state_str = ", ".join(f"{k}={v}" for k, v in state.items())
        logger.debug(f"{module_name} state: {state_str}")


def log_signal_info(logger: logging.Logger, signal_name: str, signal_data: Any):
    """
    Log signal information for debugging.
    
    Args:
        logger: Logger instance
        signal_name: Name/description of the signal
        signal_data: Signal data (will extract relevant info)
    """
    if logger.isEnabledFor(logging.DEBUG):
        if hasattr(signal_data, 'type') and hasattr(signal_data, 'value'):
            # Signal object
            logger.debug(f"{signal_name}: {signal_data.type.value}={signal_data.value}")
        elif hasattr(signal_data, '__len__'):
            # Array-like
            logger.debug(f"{signal_name}: length={len(signal_data)}")
        else:
            # Scalar value
            logger.debug(f"{signal_name}: {signal_data}")