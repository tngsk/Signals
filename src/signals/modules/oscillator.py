"""
Oscillator module for waveform generation.

This module provides the Oscillator class which generates various waveforms
including sine, square, sawtooth, triangle, and noise. The oscillator supports
frequency and amplitude modulation and maintains phase continuity.
"""

import math
from enum import Enum
from typing import Optional

import numpy as np

from ..core.module import Module, ParameterType, Signal, SignalType
from ..core.context import get_sample_rate_or_default
from ..core.logging import get_logger, performance_logger, log_module_state


class WaveformType(Enum):
    """
    Enumeration of available waveform types.

    Attributes:
        SINE: Sinusoidal waveform - smooth, fundamental frequency
        SQUARE: Square wave - rich in odd harmonics
        SAW: Sawtooth wave - rich in all harmonics, bright sound
        TRIANGLE: Triangle wave - softer than square, odd harmonics only
        NOISE: White noise - random values for percussion and effects
    """

    SINE = "sine"
    SQUARE = "square"
    SAW = "saw"
    TRIANGLE = "triangle"
    NOISE = "noise"


class Oscillator(Module):
    """
    Digital oscillator for generating various waveforms.

    The Oscillator generates audio signals with selectable waveforms and supports
    real-time parameter changes. It maintains phase continuity when parameters
    are modified and can accept frequency modulation inputs.

    Args:
        sample_rate: Audio sample rate in Hz (optional, uses context if available)
        waveform: Initial waveform type (default: SINE)

    Attributes:
        sample_rate (int): Sample rate for audio generation
        waveform (WaveformType): Current waveform type
        frequency (float): Oscillator frequency in Hz (default: 440.0)
        phase (float): Current phase position (0.0 to 1.0)
        amplitude (float): Output amplitude scaling factor (default: 1.0)

    Example:
        >>> # With explicit sample rate
        >>> osc = Oscillator(sample_rate=48000, waveform=WaveformType.SINE)
        >>> 
        >>> # Or with context (recommended)
        >>> with SynthContext(sample_rate=48000):
        ...     osc = Oscillator(waveform=WaveformType.SINE)
        >>> 
        >>> osc.set_parameter("frequency", 440.0)
        >>> osc.set_parameter("amplitude", 0.8)
        >>> signal = osc.process()[0]
    """

    def __init__(self, sample_rate: Optional[int] = None, waveform: WaveformType = WaveformType.SINE):
        super().__init__(
            input_count=1, output_count=1
        )  # Input for frequency modulation
        self.sample_rate = get_sample_rate_or_default(sample_rate)
        self.waveform = waveform
        self.frequency: float = 440.0
        self.phase: float = 0.0
        self.amplitude: float = 1.0
        self.logger = get_logger('modules.oscillator')
        
        self.logger.debug(f"Oscillator initialized: sample_rate={self.sample_rate}, waveform={waveform.value}")

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set oscillator parameters.

        Args:
            name: Parameter name. Supported parameters:
                - "frequency": Oscillator frequency in Hz
                - "amplitude": Output amplitude (0.0 to 1.0 recommended)
                - "waveform": Waveform type ("sine", "square", "saw", "triangle", "noise")
            value: Parameter value

        Note:
            Frequency and amplitude changes take effect immediately.
            Waveform changes are applied on the next process() call.
        """
        if name == "frequency":
            old_freq = self.frequency
            self.frequency = float(value)
            self.logger.debug(f"Frequency changed: {old_freq:.1f}Hz -> {self.frequency:.1f}Hz")
        elif name == "amplitude":
            old_amp = self.amplitude
            self.amplitude = float(value)
            self.logger.debug(f"Amplitude changed: {old_amp:.3f} -> {self.amplitude:.3f}")
        elif name == "waveform":
            try:
                old_waveform = self.waveform
                self.waveform = WaveformType(str(value).lower())
                self.logger.debug(f"Waveform changed: {old_waveform.value} -> {self.waveform.value}")
            except ValueError:
                self.logger.warning(f"Unknown waveform type {value}")
        else:
            self.logger.warning(f"Unknown parameter {name} for Oscillator")

    @performance_logger
    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Generate one sample of the current waveform.

        Processes the oscillator for one sample period, generating the appropriate
        waveform value and advancing the internal phase. Future versions will
        support frequency modulation via input signals.

        Args:
            inputs: Optional input signals for frequency modulation (not yet implemented)

        Returns:
            List containing one AUDIO signal with the generated sample value

        Note:
            Currently processes one sample at a time. Block processing will be
            added in future versions for improved efficiency.
        """
        # For Phase 1, we'll assume block processing isn't used yet,
        # and process one sample at a time.
        # Frequency modulation can be added later via inputs.

        if self.waveform == WaveformType.SINE:
            value = self.amplitude * math.sin(2 * math.pi * self.phase)
        elif self.waveform == WaveformType.SQUARE:
            value = self.amplitude * (1.0 if self.phase < 0.5 else -1.0)
        elif self.waveform == WaveformType.SAW:
            value = self.amplitude * (
                2.0 * (self.phase - math.floor(self.phase + 0.5))
            )  # Sawtooth from 0 to 1, then scale
        elif self.waveform == WaveformType.TRIANGLE:
            value = self.amplitude * (
                2.0 * abs(2.0 * (self.phase - math.floor(self.phase + 0.5))) - 1.0
            )
        elif self.waveform == WaveformType.NOISE:
            value = self.amplitude * (np.random.rand() * 2.0 - 1.0)
        else:
            value = 0.0

        self.phase += self.frequency / self.sample_rate
        self.phase %= 1.0  # Keep phase between 0 and 1

        return [Signal(SignalType.AUDIO, value)]
