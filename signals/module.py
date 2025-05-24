"""
Core module definitions for the signal processing framework.

This module provides the fundamental building blocks for the synthesizer:
- Base Module class for all signal processors
- Signal class for typed data containers
- Type definitions for parameters and signal types
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional
import numpy as np

#: Type alias for parameter values that can be passed to modules
ParameterType = float | int | str | bool

class SignalType(Enum):
    """
    Enumeration of signal types in the synthesizer framework.
    
    Attributes:
        AUDIO: Audio signals containing sound data
        CONTROL: Control voltage signals for parameter modulation
        TRIGGER: Gate/trigger signals for timing events
        FREQUENCY: Frequency control signals for pitch modulation
    """
    AUDIO = "audio"
    CONTROL = "control"
    TRIGGER = "trigger"
    FREQUENCY = "frequency"

class Signal:
    """
    Container for typed signal data in the synthesizer framework.
    
    A Signal encapsulates both the type of data (audio, control, etc.) and its value,
    which can be either a single sample or a block of samples.
    
    Args:
        type: The type of signal (AUDIO, CONTROL, TRIGGER, or FREQUENCY)
        value: Signal value, either a single float or numpy array of samples
        
    Attributes:
        type (SignalType): The signal type
        value (float | np.ndarray): The signal data
        
    Example:
        >>> audio_signal = Signal(SignalType.AUDIO, 0.5)
        >>> control_signal = Signal(SignalType.CONTROL, np.array([0.1, 0.2, 0.3]))
    """
    
    def __init__(self, type: SignalType, value: float | np.ndarray):
        self.type = type
        self.value = value # Can be a single float or a block of samples (np.ndarray)

    def __repr__(self):
        return f"Signal(type={self.type.value}, value_shape={np.shape(self.value)})"

class Module(ABC):
    """
    Abstract base class for all signal processing modules.
    
    This class defines the interface that all synthesizer modules must implement.
    Modules are the building blocks of the synthesis pipeline, each responsible
    for generating or processing signals.
    
    Args:
        input_count: Number of input signals this module accepts
        output_count: Number of output signals this module produces
        
    Attributes:
        input_count (int): Number of input connections
        output_count (int): Number of output connections
        
    Example:
        >>> class MyOscillator(Module):
        ...     def __init__(self):
        ...         super().__init__(input_count=0, output_count=1)
        ...     
        ...     def process(self, inputs=None):
        ...         return [Signal(SignalType.AUDIO, 0.5)]
    """
    
    def __init__(self, input_count: int, output_count: int):
        self.input_count = input_count
        self.output_count = output_count

    @abstractmethod
    def process(self, inputs: Optional[List[Signal]] = None) -> List[Signal]:
        """
        Process input signals and generate output signals.
        
        This is the core method that must be implemented by all modules.
        It takes a list of input signals and returns a list of output signals.
        
        Args:
            inputs: List of input signals, or None if no inputs
            
        Returns:
            List of output signals with length equal to output_count
            
        Note:
            The number of input signals should match input_count, and the
            number of returned signals should match output_count.
        """
        pass

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set a parameter value for this module.
        
        Parameters control the behavior of modules (e.g., frequency, amplitude).
        The base implementation prints a warning for unknown parameters.
        
        Args:
            name: Parameter name as a string
            value: Parameter value (float, int, str, or bool)
            
        Note:
            Subclasses should override this method to handle their specific parameters.
        """
        print(f"Warning: set_parameter not implemented for {self.__class__.__name__} (param: {name})")
