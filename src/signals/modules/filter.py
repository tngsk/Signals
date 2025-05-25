"""
Filter module for audio signal filtering.

This module provides the Filter class which implements various types of digital filters
including lowpass, highpass, and bandpass filters. The filters use biquad implementations
for high-quality audio processing with configurable cutoff frequency and resonance.
"""

import math
from enum import Enum
from typing import Optional

from ..core.module import Module, ParameterType, Signal, SignalType
from ..core.context import get_sample_rate_or_default
from ..core.logging import get_logger, performance_logger, log_module_state


class FilterType(Enum):
    """
    Enumeration of available filter types.

    Attributes:
        LOWPASS: Low-pass filter - allows frequencies below cutoff, attenuates above
        HIGHPASS: High-pass filter - allows frequencies above cutoff, attenuates below
        BANDPASS: Band-pass filter - allows frequencies around cutoff, attenuates others
        NOTCH: Notch filter - attenuates frequencies around cutoff, allows others
    """

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"


class Filter(Module):
    """
    Digital biquad filter for audio signal processing.

    The Filter implements various filter types using biquad filter topology for
    high-quality audio processing. It supports real-time parameter changes for
    cutoff frequency and resonance (Q factor).

    Args:
        sample_rate: Audio sample rate in Hz (optional, uses context if available)
        filter_type: Initial filter type (default: LOWPASS)

    Attributes:
        sample_rate (int): Sample rate for audio processing
        filter_type (FilterType): Current filter type
        cutoff_frequency (float): Filter cutoff frequency in Hz (default: 1000.0)
        resonance (float): Filter resonance/Q factor (default: 0.707)
        x1, x2 (float): Input delay line for biquad filter
        y1, y2 (float): Output delay line for biquad filter
        b0, b1, b2 (float): Feedforward coefficients
        a1, a2 (float): Feedback coefficients

    Example:
        >>> # With explicit sample rate
        >>> filt = Filter(sample_rate=48000, filter_type=FilterType.LOWPASS)
        >>>
        >>> # Or with context (recommended)
        >>> with SynthContext(sample_rate=48000):
        ...     filt = Filter(filter_type=FilterType.LOWPASS)
        >>>
        >>> filt.set_parameter("cutoff_frequency", 2000.0)
        >>> filt.set_parameter("resonance", 2.0)
        >>> filtered_signal = filt.process([audio_signal])[0]
    """

    def __init__(self, sample_rate: Optional[int] = None, filter_type: FilterType = FilterType.LOWPASS):
        super().__init__(input_count=1, output_count=1)  # Audio input
        self.sample_rate = get_sample_rate_or_default(sample_rate)
        self.filter_type = filter_type
        self.cutoff_frequency: float = 1000.0
        self.resonance: float = 0.707  # Q factor, 0.707 = critically damped

        # Biquad filter state variables
        self.x1: float = 0.0  # Input delay line
        self.x2: float = 0.0
        self.y1: float = 0.0  # Output delay line
        self.y2: float = 0.0

        # Biquad coefficients
        self.b0: float = 1.0  # Feedforward coefficients
        self.b1: float = 0.0
        self.b2: float = 0.0
        self.a1: float = 0.0  # Feedback coefficients (normalized, a0 = 1)
        self.a2: float = 0.0

        # Parameter smoothing
        self._target_cutoff: float = self.cutoff_frequency
        self._target_resonance: float = self.resonance
        self._smooth_factor: float = 0.995  # Smoothing coefficient
        
        # DC blocking filter
        self._dc_block_x1: float = 0.0
        self._dc_block_y1: float = 0.0
        self._dc_block_factor: float = 0.995

        self.logger = get_logger('modules.filter')

        # Calculate initial coefficients
        self._update_coefficients()

        self.logger.debug(f"Filter initialized: sample_rate={self.sample_rate}, "
                         f"type={filter_type.value}, cutoff={self.cutoff_frequency:.1f}Hz")

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set filter parameters.

        Args:
            name: Parameter name. Supported parameters:
                - "cutoff_frequency": Filter cutoff frequency in Hz
                - "resonance": Filter resonance/Q factor (0.1 to 20.0 recommended)
                - "filter_type": Filter type ("lowpass", "highpass", "bandpass", "notch")
            value: Parameter value

        Note:
            Parameter changes are smoothed to prevent clicks and trigger coefficient recalculation.
        """
        if name == "cutoff_frequency":
            old_cutoff = self._target_cutoff
            # Set target for smoothing
            nyquist = self.sample_rate / 2.0
            safe_nyquist = nyquist * 0.95  # 5%下げて安全側に
            self._target_cutoff = max(10.0, min(float(value), safe_nyquist))
            self.logger.debug(f"Target cutoff frequency set: {old_cutoff:.1f}Hz -> {self._target_cutoff:.1f}Hz")
        elif name == "resonance":
            old_q = self._target_resonance
            # Set target for smoothing
            self._target_resonance = max(0.1, min(float(value), 10.0))
            self.logger.debug(f"Target resonance set: {old_q:.3f} -> {self._target_resonance:.3f}")
        elif name == "filter_type":
            try:
                old_type = self.filter_type
                self.filter_type = FilterType(str(value).lower())
                # Force immediate coefficient update for filter type changes
                self._update_coefficients()
                self.logger.debug(f"Filter type changed: {old_type.value} -> {self.filter_type.value}")
            except ValueError:
                self.logger.warning(f"Unknown filter type {value}")
        else:
            self.logger.warning(f"Unknown parameter {name} for Filter")

    def _update_coefficients(self):
        """
        Update biquad filter coefficients based on current parameters.

        Implements standard biquad filter equations for different filter types.
        Uses bilinear transform for stable digital filter design.
        """
        # Normalize frequency (0 to π)
        omega = 2.0 * math.pi * self.cutoff_frequency / self.sample_rate
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)

        # Calculate alpha (bandwidth parameter)
        # 5. ゼロ除算対策
        resonance = max(self.resonance, 1e-5)
        alpha = sin_omega / (2.0 * resonance)

        if self.filter_type == FilterType.LOWPASS:
            # Low-pass filter coefficients
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self.filter_type == FilterType.HIGHPASS:
            # High-pass filter coefficients
            b0 = (1.0 + cos_omega) / 2.0
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self.filter_type == FilterType.BANDPASS:
            # Band-pass filter coefficients (constant skirt gain)
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        elif self.filter_type == FilterType.NOTCH:
            # Notch filter coefficients
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha

        else:
            # Default to unity gain (no filtering)
            b0, b1, b2 = 1.0, 0.0, 0.0
            a0, a1, a2 = 1.0, 0.0, 0.0

        # 5. a0の下限値設定とより安全な正規化
        a0 = max(abs(a0), 1e-10)
        if a0 < 1e-10:
            # Fallback to bypass filter if a0 is too small
            self.b0, self.b1, self.b2 = 1.0, 0.0, 0.0
            self.a1, self.a2 = 0.0, 0.0
        else:
            self.b0 = b0 / a0
            self.b1 = b1 / a0
            self.b2 = b2 / a0
            self.a1 = a1 / a0
            self.a2 = a2 / a0
            
            # Simple sanity check for extreme values
            if (abs(self.a2) > 0.999 or abs(self.a1) > 1.999 or
                any(math.isnan(x) or math.isinf(x) for x in [self.b0, self.b1, self.b2, self.a1, self.a2])):
                # Reset to bypass if coefficients are extreme
                self.b0, self.b1, self.b2 = 1.0, 0.0, 0.0
                self.a1, self.a2 = 0.0, 0.0

    @performance_logger
    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Apply digital filtering to the input audio signal.

        Processes the input through a biquad filter using the Direct Form II
        topology for numerical stability and efficiency.

        Args:
            inputs: List of input signals:
                   - inputs[0]: Audio signal to be filtered (AUDIO type)

        Returns:
            List containing one AUDIO signal with the filtered result

        Note:
            If no input is provided or input has wrong type, outputs silence.
            The filter maintains internal state for continuous processing.
        """
        # Smooth parameter changes
        self._smooth_parameters()
        
        output_value = 0.0

        if inputs and len(inputs) >= 1:
            audio_input = inputs[0]

            if audio_input.type == SignalType.AUDIO:
                x0 = audio_input.value
                
                # Apply DC blocking to input
                x0 = self._dc_block(x0)

                # Biquad filter implementation (Direct Form II)
                output_value = (self.b0 * x0 +
                               self.b1 * self.x1 +
                               self.b2 * self.x2 -
                               self.a1 * self.y1 -
                               self.a2 * self.y2)

                # NaN/Inf check
                if math.isnan(output_value) or math.isinf(output_value):
                    output_value = 0.0

                # Apply soft clipping instead of hard clipping
                output_value = self._soft_clip(output_value)

                # Update delay lines with denormal protection
                self.x2 = self.x1
                self.x1 = self._add_denormal_protection(x0)
                self.y2 = self.y1
                self.y1 = self._add_denormal_protection(output_value)

        return [Signal(SignalType.AUDIO, output_value)]

    def _smooth_parameters(self):
        """Smooth parameter changes to prevent clicks and artifacts."""
        # Only update coefficients if parameters have changed significantly
        cutoff_diff = abs(self.cutoff_frequency - self._target_cutoff)
        resonance_diff = abs(self.resonance - self._target_resonance)
        
        # Use faster smoothing for larger changes to be more responsive
        if cutoff_diff > 100.0 or resonance_diff > 0.5:
            # Faster smoothing for large changes
            smooth_factor = 0.9
        elif cutoff_diff > 1.0 or resonance_diff > 0.01:
            # Normal smoothing
            smooth_factor = self._smooth_factor
        else:
            # No need to update if changes are very small
            return
            
        self.cutoff_frequency = (self.cutoff_frequency * smooth_factor + 
                               self._target_cutoff * (1.0 - smooth_factor))
        self.resonance = (self.resonance * smooth_factor + 
                        self._target_resonance * (1.0 - smooth_factor))
        self._update_coefficients()
    
    def _add_denormal_protection(self, value: float) -> float:
        """Add denormal protection to prevent very small values that can cause performance issues."""
        if abs(value) < 1e-20:
            return 0.0
        return value
    
    def _dc_block(self, input_value: float) -> float:
        """Simple DC blocking filter to remove DC offset."""
        output = input_value - self._dc_block_x1 + self._dc_block_factor * self._dc_block_y1
        self._dc_block_x1 = input_value
        self._dc_block_y1 = output
        return output
    

    
    def _soft_clip(self, value: float, threshold: float = 0.8) -> float:
        """Apply soft clipping for more musical saturation instead of hard clipping."""
        if abs(value) <= threshold:
            return value
        
        sign = 1.0 if value >= 0 else -1.0
        excess = abs(value) - threshold
        compressed = threshold + excess / (1.0 + excess)
        return sign * min(compressed, 1.0)

    def reset(self):
        """
        Reset filter state variables.

        Clears the internal delay lines, useful when starting a new audio stream
        or when discontinuities might cause artifacts.
        """
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0
        self._dc_block_x1 = self._dc_block_y1 = 0.0
        self.logger.debug("Filter state reset")
