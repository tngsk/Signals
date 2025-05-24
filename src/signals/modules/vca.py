"""
Voltage Controlled Amplifier (VCA) module for amplitude modulation.

This module provides the VCA class which multiplies an audio input signal
by a control voltage signal, enabling dynamic amplitude control through
envelopes, LFOs, or other modulation sources.
"""

from ..core.module import Module, ParameterType, Signal, SignalType


class VCA(Module):
    """
    Voltage Controlled Amplifier for amplitude modulation.

    The VCA multiplies an audio input signal by a control voltage signal,
    allowing for dynamic amplitude control. This is a fundamental building
    block in modular synthesis for applying envelopes, tremolo effects,
    and other amplitude-based modulations.

    Args:
        sample_rate: Audio sample rate in Hz (required for consistency with other modules)

    Attributes:
        gain (float): Base gain level applied before control voltage modulation (default: 1.0)

    Example:
        >>> vca = VCA()
        >>> vca.set_parameter("gain", 0.8)
        >>> # Connect audio to input 0, control voltage to input 1
        >>> modulated_signal = vca.process([audio_signal, cv_signal])[0]
    """

    def __init__(self, sample_rate: int):
        super().__init__(input_count=2, output_count=1)  # Audio input + CV input
        self.sample_rate = sample_rate
        self.gain: float = 1.0

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set VCA parameters.

        Args:
            name: Parameter name. Supported parameters:
                - "gain": Base gain level (0.0 to 1.0 recommended, but can exceed for amplification)
            value: Parameter value

        Note:
            The gain parameter sets the base level before control voltage modulation.
            The final output is: input_audio * gain * control_voltage
        """
        if name == "gain":
            self.gain = float(value)
        else:
            print(f"Warning: Unknown parameter {name} for VCA")

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Apply voltage controlled amplification to the input audio signal.

        Multiplies the audio input by the control voltage input and base gain.
        If no control voltage is provided, uses the base gain only.

        Args:
            inputs: List of input signals:
                   - inputs[0]: Audio signal to be amplified (AUDIO type)
                   - inputs[1]: Control voltage signal (CONTROL or AUDIO type)

        Returns:
            List containing one AUDIO signal with the amplified result

        Note:
            If either input is missing or has the wrong type, the output will be silence.
            Control voltage values typically range from 0.0 to 1.0, but can exceed this range.
        """
        output_value = 0.0

        if inputs and len(inputs) >= 2:
            audio_input = inputs[0]
            cv_input = inputs[1]

            # Process only if we have proper signal types
            if (audio_input.type == SignalType.AUDIO and 
                cv_input.type in [SignalType.CONTROL, SignalType.AUDIO]):
                
                output_value = audio_input.value * cv_input.value * self.gain
                
        elif inputs and len(inputs) == 1:
            # If only audio input provided, apply base gain only
            audio_input = inputs[0]
            if audio_input.type == SignalType.AUDIO:
                output_value = audio_input.value * self.gain

        return [Signal(SignalType.AUDIO, output_value)]