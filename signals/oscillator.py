from .module import Module, Signal, SignalType, ParameterType
from enum import Enum
import numpy as np
import math

class WaveformType(Enum):
    SINE = "sine"
    SQUARE = "square"
    SAW = "saw"
    TRIANGLE = "triangle"
    NOISE = "noise"

class Oscillator(Module):
    def __init__(self, sample_rate: int, waveform: WaveformType = WaveformType.SINE):
        super().__init__(input_count=1, output_count=1) # Input for frequency modulation
        self.sample_rate = sample_rate
        self.waveform = waveform
        self.frequency: float = 440.0
        self.phase: float = 0.0
        self.amplitude: float = 1.0

    def set_parameter(self, name: str, value: ParameterType):
        if name == "frequency":
            self.frequency = float(value)
        elif name == "amplitude":
            self.amplitude = float(value)
        elif name == "waveform":
            try:
                self.waveform = WaveformType(str(value).lower())
            except ValueError:
                print(f"Warning: Unknown waveform type {value}")
        else:
            print(f"Warning: Unknown parameter {name} for Oscillator")

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        # For Phase 1, we'll assume block processing isn't used yet,
        # and process one sample at a time.
        # Frequency modulation can be added later via inputs.

        if self.waveform == WaveformType.SINE:
            value = self.amplitude * math.sin(2 * math.pi * self.phase)
        elif self.waveform == WaveformType.SQUARE:
            value = self.amplitude * (1.0 if self.phase < 0.5 else -1.0)
        elif self.waveform == WaveformType.SAW:
            value = self.amplitude * (2.0 * (self.phase - math.floor(self.phase + 0.5))) # Sawtooth from 0 to 1, then scale
        elif self.waveform == WaveformType.TRIANGLE:
            value = self.amplitude * (2.0 * abs(2.0 * (self.phase - math.floor(self.phase + 0.5))) - 1.0)
        elif self.waveform == WaveformType.NOISE:
            value = self.amplitude * (np.random.rand() * 2.0 - 1.0)
        else:
            value = 0.0

        self.phase += self.frequency / self.sample_rate
        self.phase %= 1.0  # Keep phase between 0 and 1

        return [Signal(SignalType.AUDIO, value)]
