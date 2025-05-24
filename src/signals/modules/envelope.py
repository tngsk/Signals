"""
Envelope generator module for amplitude and parameter modulation.

This module provides ADSR (Attack-Decay-Sustain-Release) envelope generation
for controlling amplitude, filter cutoff, and other time-varying parameters
in the synthesizer. The envelope responds to trigger signals and provides
smooth transitions between different phases.
"""

import numpy as np
import re
from typing import Union

from ..core.module import Module, ParameterType, Signal, SignalType
from ..core.logging import get_logger, performance_logger, log_module_state


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
        self.logger = get_logger('modules.envelope')
        
        # Original time-based parameters
        self.attack_time: float = 0.05  # seconds
        self.decay_time: float = 0.1  # seconds
        self.sustain_level: float = 0.7  # 0.0 to 1.0
        self.release_time: float = 0.2  # seconds
        
        # Enhanced parameter storage (raw values as entered by user)
        self._attack_param: Union[str, float] = 0.05
        self._decay_param: Union[str, float] = 0.1
        self._release_param: Union[str, float] = 0.2
        
        # Total duration for relative calculations (set externally or auto-detected)
        self._total_duration: float = 2.0  # Default duration
        
        # Auto mode calculation flags
        self._auto_attack: bool = False
        self._auto_decay: bool = False
        self._auto_release: bool = False

        self._attack_samples: int = int(self.attack_time * self.sample_rate)
        self._decay_samples: int = int(self.decay_time * self.sample_rate)
        self._release_samples: int = int(self.release_time * self.sample_rate)

        self._phase: int = 0  # 0: idle, 1: attack, 2: decay, 3: sustain, 4: release
        self._current_sample: int = 0
        self._value: float = 0.0
        self._note_on: bool = False
        
        self.logger.debug(f"EnvelopeADSR initialized: sample_rate={sample_rate}")

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set envelope parameters with enhanced support for relative values and auto mode.

        Args:
            name: Parameter name. Supported parameters:
                - "attack": Attack time - supports multiple formats:
                  * Seconds: 0.05 (any float/int value)
                  * Percentage: "5%" (percentage of total duration)
                  * Auto: "auto" (calculated from other parameters and total duration)
                - "decay": Decay time (same formats as attack)
                - "sustain": Sustain level (0.0 to 1.0)
                - "release": Release time (same formats as attack)
                - "total_duration": Set total duration for relative calculations
            value: Parameter value in various formats

        Examples:
            >>> env.set_parameter("attack", "5%")      # 5% of total duration
            >>> env.set_parameter("decay", "auto")     # Auto-calculated
            >>> env.set_parameter("release", 1.5)      # 1.5 seconds
            >>> env.set_parameter("attack", 0.05)      # 0.05 seconds
        """
        if name == "attack":
            self._attack_param = value
            self._auto_attack = (value == "auto")
            if not self._auto_attack:
                self.attack_time = self._parse_time_parameter(value)
                self._attack_samples = int(self.attack_time * self.sample_rate)
            self._recalculate_auto_times()
            self.logger.debug(f"Attack parameter set: {value} -> {self.attack_time:.3f}s")
        elif name == "decay":
            self._decay_param = value
            self._auto_decay = (value == "auto")
            if not self._auto_decay:
                self.decay_time = self._parse_time_parameter(value)
                self._decay_samples = int(self.decay_time * self.sample_rate)
            self._recalculate_auto_times()
            self.logger.debug(f"Decay parameter set: {value} -> {self.decay_time:.3f}s")
        elif name == "sustain":
            self.sustain_level = float(value)
            self.logger.debug(f"Sustain level set: {value}")
        elif name == "release":
            self._release_param = value
            self._auto_release = (value == "auto")
            if not self._auto_release:
                self.release_time = self._parse_time_parameter(value)
                self._release_samples = int(self.release_time * self.sample_rate)
            self._recalculate_auto_times()
            self.logger.debug(f"Release parameter set: {value} -> {self.release_time:.3f}s")
        elif name == "total_duration":
            self._total_duration = float(value)
            # Recalculate all relative parameters
            self._recalculate_times()
            self.logger.info(f"Total duration set: {value}s, recalculating envelope times")
        else:
            self.logger.warning(f"Unknown parameter {name} for EnvelopeADSR")

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
        self.logger.debug("Envelope triggered ON - starting attack phase")

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
            self.logger.debug(f"Envelope triggered OFF - starting release phase from value {self._value:.3f}")

    @performance_logger
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
    
    def _parse_time_parameter(self, value: Union[str, float]) -> float:
        """
        Parse a time parameter that can be in various formats.
        
        Args:
            value: Parameter value (string or number)
            
        Returns:
            Time in seconds
        """
        if isinstance(value, str):
            value_str = value.strip()
            
            # Percentage format (e.g., "5%", "20%")
            if value_str.endswith('%'):
                try:
                    percentage = float(value_str[:-1])
                    return self._total_duration * (percentage / 100.0)
                except ValueError:
                    self.logger.warning(f"Invalid percentage format '{value}', using 0.1 seconds")
                    return 0.1
            
            # Try to parse as plain number string (treat as seconds)
            try:
                return float(value_str)
            except ValueError:
                self.logger.warning(f"Unable to parse time parameter '{value}', using 0.1 seconds")
                return 0.1
        
        elif isinstance(value, (int, float)):
            # All numeric values are treated as seconds
            return float(value)
        
        else:
            self.logger.warning(f"Invalid time parameter type for '{value}', using 0.1 seconds")
            return 0.1
    
    def _recalculate_times(self):
        """Recalculate all time parameters when total_duration changes."""
        if not self._auto_attack:
            self.attack_time = self._parse_time_parameter(self._attack_param)
            self._attack_samples = int(self.attack_time * self.sample_rate)
        
        if not self._auto_decay:
            self.decay_time = self._parse_time_parameter(self._decay_param)
            self._decay_samples = int(self.decay_time * self.sample_rate)
        
        if not self._auto_release:
            self.release_time = self._parse_time_parameter(self._release_param)
            self._release_samples = int(self.release_time * self.sample_rate)
        
        self._recalculate_auto_times()
    
    def _recalculate_auto_times(self):
        """Recalculate auto parameters based on total duration and other set parameters."""
        # Calculate the total used time by non-auto parameters
        used_time = 0.0
        auto_count = 0
        
        if not self._auto_attack:
            used_time += self.attack_time
        else:
            auto_count += 1
        
        if not self._auto_decay:
            used_time += self.decay_time
        else:
            auto_count += 1
        
        if not self._auto_release:
            used_time += self.release_time
        else:
            auto_count += 1
        
        # Calculate remaining time for auto parameters
        if auto_count > 0:
            # Reserve some time for sustain (minimum 10% of total duration)
            reserved_sustain_time = max(0.1 * self._total_duration, 0.1)
            available_time = max(0.0, self._total_duration - used_time - reserved_sustain_time)
            auto_time_each = available_time / auto_count if auto_count > 0 else 0.0
            
            # Ensure minimum time for each auto parameter
            auto_time_each = max(auto_time_each, 0.01)  # Minimum 10ms
            

            
            if self._auto_attack:
                self.attack_time = auto_time_each
                self._attack_samples = int(self.attack_time * self.sample_rate)
            
            if self._auto_decay:
                self.decay_time = auto_time_each
                self._decay_samples = int(self.decay_time * self.sample_rate)
            
            if self._auto_release:
                self.release_time = auto_time_each
                self._release_samples = int(self.release_time * self.sample_rate)
    
    def set_total_duration(self, duration: float):
        """
        Set the total duration for relative calculations.
        
        Args:
            duration: Total duration in seconds
        """
        self._total_duration = duration
        self._recalculate_times()
        
        # Log the final envelope configuration
        total_envelope_time = self.attack_time + self.decay_time + self.release_time
        self.logger.info(f"Envelope timing configured - Total: {duration:.3f}s, "
                        f"A:{self.attack_time:.3f}s, D:{self.decay_time:.3f}s, "
                        f"S:{self.sustain_level:.2f}, R:{self.release_time:.3f}s, "
                        f"ADSR_total: {total_envelope_time:.3f}s")
        
        if total_envelope_time > duration:
            self.logger.warning(f"ADSR total ({total_envelope_time:.3f}s) exceeds "
                              f"duration ({duration:.3f}s) by {total_envelope_time - duration:.3f}s")
    
    def get_info(self) -> dict:
        """
        Get current envelope information including calculated times.
        
        Returns:
            Dictionary with envelope state and calculated values
        """
        return {
            'attack_time': self.attack_time,
            'decay_time': self.decay_time,
            'sustain_level': self.sustain_level,
            'release_time': self.release_time,
            'total_duration': self._total_duration,
            'attack_param': self._attack_param,
            'decay_param': self._decay_param,
            'release_param': self._release_param,
            'auto_attack': self._auto_attack,
            'auto_decay': self._auto_decay,
            'auto_release': self._auto_release,
            'current_phase': self._phase,
            'current_value': self._value
        }
