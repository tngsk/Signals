from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional
import numpy as np

ParameterType = float | int | str | bool

class SignalType(Enum):
    AUDIO = "audio"
    CONTROL = "control"
    TRIGGER = "trigger"
    FREQUENCY = "frequency"

class Signal:
    def __init__(self, type: SignalType, value: float | np.ndarray):
        self.type = type
        self.value = value # Can be a single float or a block of samples (np.ndarray)

    def __repr__(self):
        return f"Signal(type={self.type.value}, value_shape={np.shape(self.value)})"

class Module(ABC):
    def __init__(self, input_count: int, output_count: int):
        self.input_count = input_count
        self.output_count = output_count

    @abstractmethod
    def process(self, inputs: Optional[List[Signal]] = None) -> List[Signal]:
        pass

    def set_parameter(self, name: str, value: ParameterType):
        print(f"Warning: set_parameter not implemented for {self.__class__.__name__} (param: {name})")
