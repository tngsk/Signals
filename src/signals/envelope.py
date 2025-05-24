"""
Envelope generator module for amplitude and parameter modulation.

This module provides ADSR (Attack-Decay-Sustain-Release) envelope generation
for controlling amplitude, filter cutoff, and other time-varying parameters
in the synthesizer. The envelope responds to trigger signals and provides
smooth transitions between different phases.
"""

import numpy as np

from .module import Module, ParameterType, Signal, SignalType


class EnvelopeADSR(Module):
    """
    ADSR (Attack-Decay-Sustain-Release) envelope generator.

    Generates control voltage signals that follow the classic ADSR envelope shape:
    - Attack: Linear rise from 0 to 1
    - Decay: Linear fall from 1 to sustain level
    - Sustain: Constant level while gate is held
    - Release: Linear fall from sustain level to 0

    Args:
        sample_rate: Audio sample rate in Hz

    Attributes:
        sample_rate (int): Sample rate for timing calculations
        attack_time (float): Attack phase duration in seconds (default: 0.05)
        decay_time (float): Decay phase duration in seconds (default: 0.1)
        sustain_level (float): Sustain level (0.0 to 1.0, default: 0.7)
        release_time (float): Release phase duration in seconds (default: 0.2)

    Example:
        >>> env = EnvelopeADSR(sample_rate=48000)
        >>> env.set_parameter("attack", 0.1)
        >>> env.set_parameter("sustain", 0.6)
        >>> env.trigger_on()  # Start envelope
        >>> signal = env.process()[0]  # Get envelope value
        >>> env.trigger_off()  # Begin release phase
    """

    def __init__(self, sample_rate: int):
        super().__init__(input_count=1, output_count=1)  # Input for trigger
        self.sample_rate = sample_rate
        self.attack_time: float = 0.05  # seconds
        self.decay_time: float = 0.1  # seconds
        self.sustain_level: float = 0.7  # 0.0 to 1.0
        self.release_time: float = 0.2  # seconds

        self._attack_samples: int = int(self.attack_time * self.sample_rate)
        self._decay_samples: int = int(self.decay_time * self.sample_rate)
        self._release_samples: int = int(self.release_time * self.sample_rate)

        self._phase: int = 0  # 0: idle, 1: attack, 2: decay, 3: sustain, 4: release
        self._current_sample: int = 0
        self._value: float = 0.0
        self._note_on: bool = False

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set envelope parameters.

        Args:
            name: Parameter name. Supported parameters:
                - "attack": Attack time in seconds (>= 0.0)
                - "decay": Decay time in seconds (>= 0.0)
                - "sustain": Sustain level (0.0 to 1.0)
                - "release": Release time in seconds (>= 0.0)
            value: Parameter value

        Note:
            Parameter changes take effect immediately and may cause
            audible clicks if changed during active envelope phases.
        """
        if name == "attack":
            self.attack_time = float(value)
            self._attack_samples = int(self.attack_time * self.sample_rate)
        elif name == "decay":
            self.decay_time = float(value)
            self._decay_samples = int(self.decay_time * self.sample_rate)
        elif name == "sustain":
            self.sustain_level = float(value)
        elif name == "release":
            self.release_time = float(value)
            self._release_samples = int(self.release_time * self.sample_rate)
        else:
            print(f"Warning: Unknown parameter {name} for EnvelopeADSR")

    def trigger_on(self):
        """
        Trigger the envelope to start the attack phase.

        Resets the envelope to the beginning of the attack phase,
        regardless of the current state. This simulates pressing
        a key on a keyboard or receiving a MIDI note-on event.
        """
        self._note_on = True
        self._phase = 1  # Attack
        self._current_sample = 0

    def trigger_off(self):
        """
        Release the envelope to start the release phase.

        Transitions the envelope to the release phase, causing it
        to decay from the current level to zero. This simulates
        releasing a key or receiving a MIDI note-off event.

        Note:
            If the envelope is already in the release or idle phase,
            this method has no effect.
        """
        self._note_on = False
        if self._phase != 0:  # If not idle
            self._phase = 4  # Release
            self._current_sample = 0

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Process the envelope for one sample period.

        Advances the envelope state machine by one sample and calculates
        the current envelope value. Can respond to trigger input signals
        or be controlled manually via trigger_on/trigger_off methods.

        Args:
            inputs: Optional list of input signals. If provided, the first
                   signal should be a TRIGGER type where:
                   - Value > 0.5: Trigger on (start attack)
                   - Value <= 0.5: Trigger off (start release)

        Returns:
            List containing one CONTROL signal with the current envelope value (0.0 to 1.0)

        Note:
            The envelope value is clipped to the range [0.0, 1.0] to ensure
            stable behavior when used as a control voltage.
        """
        if (
            inputs
            and inputs[0].type == SignalType.TRIGGER
            and inputs[0].value > 0.5
            and not self._note_on
        ):
            self.trigger_on()
        elif (
            inputs
            and inputs[0].type == SignalType.TRIGGER
            and inputs[0].value <= 0.5
            and self._note_on
        ):
            self.trigger_off()

        if self._phase == 1:  # Attack
            self._value = (
                self._current_sample / self._attack_samples
                if self._attack_samples > 0
                else 1.0
            )
            if self._current_sample >= self._attack_samples:
                self._phase = 2  # Decay
                self._current_sample = 0
        elif self._phase == 2:  # Decay
            progress = (
                self._current_sample / self._decay_samples
                if self._decay_samples > 0
                else 1.0
            )
            self._value = 1.0 - (1.0 - self.sustain_level) * progress
            if self._current_sample >= self._decay_samples:
                self._phase = 3  # Sustain
        elif self._phase == 3:  # Sustain
            self._value = self.sustain_level
            if (
                not self._note_on
            ):  # Should have been caught by trigger_off, but as a safeguard
                self._phase = 4  # Release
                self._current_sample = 0
        elif self._phase == 4:  # Release
            self._value = self.sustain_level * (
                1.0
                - (
                    self._current_sample / self._release_samples
                    if self._release_samples > 0
                    else 1.0
                )
            )
            if self._current_sample >= self._release_samples:
                self._phase = 0  # Idle
                self._value = 0.0

        if self._phase != 0:
            self._current_sample += 1

        return [Signal(SignalType.CONTROL, np.clip(self._value, 0.0, 1.0))]
