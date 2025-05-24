"""
Synthesis context management for centralized sample rate and configuration.

This module provides the SynthContext class which manages global synthesis
parameters like sample rate, enabling modules to be created without explicit
sample rate specification while maintaining consistency across the synthesis graph.
"""

import threading
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager

from .logging import get_logger


class ContextError(Exception):
    """Exception raised for context-related errors."""
    pass


class SynthContext:
    """
    Thread-safe context manager for synthesis parameters.
    
    Provides centralized management of sample rate and other synthesis
    parameters, allowing modules to be created without explicit parameter
    specification while ensuring consistency across the synthesis graph.
    
    Example:
        >>> with SynthContext(sample_rate=48000):
        ...     osc = Oscillator()  # Uses 48000 Hz automatically
        ...     env = EnvelopeADSR()  # Uses 48000 Hz automatically
        
        >>> # Or using SynthEngine context
        >>> engine = SynthEngine(sample_rate=48000)
        >>> with engine.context():
        ...     osc = Oscillator()  # Uses engine's sample rate
    """
    
    # Thread-local storage for context data
    _local = threading.local()
    _logger = get_logger('core.context')
    
    def __init__(self, sample_rate: int, **kwargs):
        """
        Initialize synthesis context.
        
        Args:
            sample_rate: Audio sample rate in Hz
            **kwargs: Additional context parameters for future expansion
        """
        self.sample_rate = sample_rate
        self.parameters = kwargs
        self._logger.debug(f"SynthContext created: sample_rate={sample_rate}, params={kwargs}")
    
    def __enter__(self):
        """Enter the context manager."""
        # Store previous context for nesting support
        previous_context = getattr(self._local, 'context_stack', [])
        previous_context.append(getattr(self._local, 'current_context', None))
        
        self._local.context_stack = previous_context
        self._local.current_context = self
        
        self._logger.debug(f"Entered SynthContext: sample_rate={self.sample_rate}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Restore previous context
        context_stack = getattr(self._local, 'context_stack', [])
        if context_stack:
            self._local.current_context = context_stack.pop()
        else:
            self._local.current_context = None
        
        self._logger.debug(f"Exited SynthContext: sample_rate={self.sample_rate}")
    
    @classmethod
    def get_current(cls) -> 'SynthContext':
        """
        Get the current active synthesis context.
        
        Returns:
            Current SynthContext instance
            
        Raises:
            ContextError: If no context is currently active
        """
        current = getattr(cls._local, 'current_context', None)
        if current is None:
            raise ContextError(
                "No synthesis context available. "
                "Create modules within a SynthContext or SynthEngine context."
            )
        return current
    
    @classmethod
    def get_sample_rate(cls) -> int:
        """
        Get the sample rate from the current context.
        
        Returns:
            Sample rate in Hz
            
        Raises:
            ContextError: If no context is currently active
        """
        return cls.get_current().sample_rate
    
    @classmethod
    def get_parameter(cls, name: str, default: Any = None) -> Any:
        """
        Get a parameter from the current context.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
            
        Raises:
            ContextError: If no context is currently active
        """
        context = cls.get_current()
        return context.parameters.get(name, default)
    
    @classmethod
    def has_context(cls) -> bool:
        """
        Check if a synthesis context is currently active.
        
        Returns:
            True if context is available, False otherwise
        """
        return getattr(cls._local, 'current_context', None) is not None
    
    @classmethod
    def get_context_info(cls) -> Dict[str, Any]:
        """
        Get information about the current context.
        
        Returns:
            Dictionary with context information
            
        Raises:
            ContextError: If no context is currently active
        """
        context = cls.get_current()
        return {
            'sample_rate': context.sample_rate,
            'parameters': context.parameters.copy(),
            'thread_id': threading.get_ident(),
            'context_depth': len(getattr(cls._local, 'context_stack', []))
        }
    
    @classmethod
    @contextmanager
    def temporary(cls, sample_rate: int, **kwargs) -> ContextManager['SynthContext']:
        """
        Create a temporary context for a specific operation.
        
        Args:
            sample_rate: Sample rate for the temporary context
            **kwargs: Additional parameters
            
        Yields:
            SynthContext instance
            
        Example:
            >>> with SynthContext.temporary(44100):
            ...     osc = Oscillator()  # Uses 44100 Hz
        """
        context = cls(sample_rate, **kwargs)
        with context:
            yield context


def get_sample_rate_or_default(sample_rate: Optional[int] = None, 
                              default: int = 48000) -> int:
    """
    Get sample rate from context or use provided/default value.
    
    This utility function provides a clean way for modules to handle
    sample rate specification with the following priority:
    1. Explicitly provided sample_rate parameter
    2. Current synthesis context sample rate
    3. Default value
    
    Args:
        sample_rate: Explicitly provided sample rate (highest priority)
        default: Default sample rate if no context available
        
    Returns:
        Sample rate to use
        
    Example:
        >>> # In module __init__:
        >>> def __init__(self, sample_rate: Optional[int] = None):
        ...     self.sample_rate = get_sample_rate_or_default(sample_rate)
    """
    if sample_rate is not None:
        return sample_rate
    
    try:
        return SynthContext.get_sample_rate()
    except ContextError:
        SynthContext._logger.debug(f"No context available, using default sample rate: {default}")
        return default


def require_context() -> SynthContext:
    """
    Require that a synthesis context is active.
    
    Returns:
        Current SynthContext
        
    Raises:
        ContextError: If no context is active
        
    Example:
        >>> def some_function():
        ...     context = require_context()
        ...     # Function requires being called within a context
    """
    return SynthContext.get_current()


@contextmanager
def synthesis_context(sample_rate: int, **kwargs) -> ContextManager[SynthContext]:
    """
    Convenience function to create a synthesis context.
    
    Args:
        sample_rate: Sample rate for the context
        **kwargs: Additional context parameters
        
    Yields:
        SynthContext instance
        
    Example:
        >>> with synthesis_context(48000) as ctx:
        ...     osc = Oscillator()
        ...     print(f"Using sample rate: {ctx.sample_rate}")
    """
    with SynthContext(sample_rate, **kwargs) as context:
        yield context