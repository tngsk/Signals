"""
LFO (Low Frequency Oscillator) module for modulation purposes.

This module provides the LFO class which generates low-frequency control signals
for modulating other synthesizer parameters. Unlike audio oscillators, LFOs output
CONTROL signals in the sub-audio range (typically 0.1-20 Hz) and are designed
specifically for parameter modulation rather than audio synthesis.
"""

import math
from typing import Optional

from ..core.module import Module, ParameterType, Signal, SignalType
from .oscillator import WaveformType
from ..core.context import get_sample_rate_or_default
from ..core.logging import get_logger, performance_logger


class LFO(Module):
    """
    Low Frequency Oscillator for generating modulation control signals.

    The LFO generates low-frequency waveforms primarily intended for modulating
    other synthesizer parameters such as amplitude (tremolo), frequency (vibrato),
    filter cutoff, or any other modulatable parameter. It supports various waveforms
    and includes phase control and retriggering capabilities.

    Args:
        sample_rate: Audio sample rate in Hz (optional, uses context if available)
        waveform: Initial waveform type (default: SINE)

    Attributes:
        sample_rate (int): Sample rate for processing
        waveform (WaveformType): Current waveform type
        frequency (float): LFO frequency in Hz (default: 1.0)
        amplitude (float): LFO amplitude 0.0-1.0 (default: 1.0)
        phase_offset (float): Phase offset in degrees 0.0-360.0 (default: 0.0)
        phase (float): Current phase accumulator
        phase_increment (float): Phase increment per sample

    Example:
        >>> # With explicit sample rate
        >>> lfo = LFO(sample_rate=48000, waveform=WaveformType.SINE)
        >>>
        >>> # Or with context (recommended)
        >>> with SynthContext(sample_rate=48000):
        ...     lfo = LFO(waveform=WaveformType.TRIANGLE)
        >>>
        >>> lfo.set_parameter("frequency", 2.0)  # 2 Hz
        >>> lfo.set_parameter("amplitude", 0.8)
        >>> control_signal = lfo.process()[0]  # Returns CONTROL signal
        >>>
        >>> # Use for modulation
        >>> vca = VCA()
        >>> audio_in = osc.process()[0]
        >>> modulated = vca.process([audio_in, control_signal])[0]
    """

    def __init__(self, sample_rate: Optional[int] = None, waveform: WaveformType = WaveformType.SINE):
        super().__init__(input_count=1, output_count=1)  # Optional trigger input
        self.sample_rate = get_sample_rate_or_default(sample_rate)
        self.waveform = waveform
        
        # LFO parameters
        self.frequency: float = 1.0  # 1 Hz default
        self.amplitude: float = 1.0  # Full amplitude
        self.phase_offset: float = 0.0  # No phase offset
        
        # Internal state
        self.phase: float = 0.0  # Current phase (0.0 to 1.0)
        self.phase_increment: float = 0.0
        
        self.logger = get_logger('modules.lfo')
        
        # Calculate initial phase increment
        self._update_phase_increment()
        
        self.logger.debug(f"LFO initialized: sample_rate={self.sample_rate}, "
                         f"waveform={waveform.value}, frequency={self.frequency:.2f}Hz")

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set LFO parameters.

        Args:
            name: Parameter name. Supported parameters:
                - "frequency": LFO frequency in Hz (0.01 to 50.0 recommended)
                - "amplitude": LFO amplitude (0.0 to 1.0)
                - "phase_offset": Phase offset in degrees (0.0 to 360.0)
                - "waveform": Waveform type (WaveformType enum or string)
            value: Parameter value

        Note:
            Frequency changes update the phase increment for the next processing cycle.
            Phase offset is applied as a constant offset to the generated waveform.
        """
        if name == "frequency":
            old_freq = self.frequency
            # Allow wider range for LFO: 0.01 Hz to 50 Hz
            self.frequency = max(0.01, min(float(value), 50.0))
            self._update_phase_increment()
            self.logger.debug(f"LFO frequency changed: {old_freq:.3f}Hz -> {self.frequency:.3f}Hz")
            
        elif name == "amplitude":
            old_amp = self.amplitude
            self.amplitude = max(0.0, min(float(value), 1.0))
            self.logger.debug(f"LFO amplitude changed: {old_amp:.3f} -> {self.amplitude:.3f}")
            
        elif name == "phase_offset":
            old_offset = self.phase_offset
            # Normalize to 0-360 range
            self.phase_offset = float(value) % 360.0
            self.logger.debug(f"LFO phase offset changed: {old_offset:.1f}° -> {self.phase_offset:.1f}°")
            
        elif name == "waveform":
            try:
                old_waveform = self.waveform
                if isinstance(value, WaveformType):
                    self.waveform = value
                else:
                    # WaveformType values are lowercase, so convert string to lowercase
                    str_value = str(value).lower()
                    # Find the enum member by value
                    for wf in WaveformType:
                        if wf.value == str_value:
                            self.waveform = wf
                            break
                    else:
                        raise ValueError(f"Unknown waveform: {value}")
                self.logger.debug(f"LFO waveform changed: {old_waveform.value} -> {self.waveform.value}")
            except ValueError:
                self.logger.warning(f"Unknown waveform type {value} for LFO")
        else:
            self.logger.warning(f"Unknown parameter {name} for LFO")

    def _update_phase_increment(self):
        """Update phase increment based on current frequency and sample rate."""
        self.phase_increment = self.frequency / self.sample_rate

    def _generate_waveform(self, phase: float) -> float:
        """
        Generate waveform sample for given phase.

        Args:
            phase: Phase value (0.0 to 1.0)

        Returns:
            Waveform sample value (-1.0 to 1.0)
        """
        if self.waveform == WaveformType.SINE:
            return math.sin(2.0 * math.pi * phase)
            
        elif self.waveform == WaveformType.SQUARE:
            return 1.0 if phase < 0.5 else -1.0
            
        elif self.waveform == WaveformType.SAW:
            return 2.0 * phase - 1.0
            
        elif self.waveform == WaveformType.TRIANGLE:
            if phase < 0.5:
                return 4.0 * phase - 1.0
            else:
                return 3.0 - 4.0 * phase
                
        elif self.waveform == WaveformType.NOISE:
            # Use simple pseudo-random based on phase for consistency
            # This ensures repeatable noise pattern at same frequency
            import random
            random.seed(int(phase * 1000000))
            return random.uniform(-1.0, 1.0)
            
        else:
            return 0.0

    @performance_logger
    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Generate LFO control signal.

        Processes the LFO to generate a low-frequency control signal based on
        the current waveform, frequency, amplitude, and phase settings.

        Args:
            inputs: Optional list of input signals:
                   - inputs[0]: Trigger signal for phase reset (TRIGGER type)

        Returns:
            List containing one CONTROL signal with the LFO output (-amplitude to +amplitude)

        Note:
            If a trigger input is provided and is active (> 0.5), the LFO phase
            will be reset to the phase offset position.
        """
        # Check for trigger input (phase reset)
        if inputs and len(inputs) >= 1:
            trigger_input = inputs[0]
            if trigger_input.type == SignalType.TRIGGER and trigger_input.value > 0.5:
                # Reset phase to phase offset
                self.phase = (self.phase_offset / 360.0) % 1.0
                self.logger.debug("LFO phase reset by trigger")

        # Apply phase offset
        current_phase = (self.phase + (self.phase_offset / 360.0)) % 1.0
        
        # Generate waveform sample
        waveform_value = self._generate_waveform(current_phase)
        
        # Apply amplitude scaling
        output_value = waveform_value * self.amplitude
        
        # Advance phase
        self.phase = (self.phase + self.phase_increment) % 1.0
        
        return [Signal(SignalType.CONTROL, output_value)]

    def reset(self):
        """
        Reset LFO state.

        Resets the phase accumulator to the phase offset position, effectively
        restarting the LFO cycle. Useful when starting a new sequence or when
        synchronization is required.
        """
        self.phase = (self.phase_offset / 360.0) % 1.0
        self.logger.debug("LFO state reset")

    def get_info(self) -> dict:
        """
        Get current LFO information.

        Returns:
            Dictionary containing current LFO state and parameters
        """
        return {
            "module_type": "LFO",
            "waveform": self.waveform.value,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "phase_offset": self.phase_offset,
            "current_phase": self.phase,
            "phase_increment": self.phase_increment,
            "sample_rate": self.sample_rate
        }