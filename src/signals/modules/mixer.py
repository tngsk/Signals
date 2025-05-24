"""
Mixer module for combining multiple audio signals.

This module provides the Mixer class which combines multiple input audio signals
into a single output signal. Each input channel has an independent gain control
for level adjustment and creative mixing effects.
"""

from ..core.module import Module, ParameterType, Signal, SignalType


class Mixer(Module):
    """
    Multi-channel audio mixer with per-channel gain control.

    The Mixer combines multiple audio input signals into a single output signal.
    Each input channel has an independent gain parameter that can be adjusted
    to control the level and balance of the mix. Channels are summed together
    with their respective gain values applied.

    Args:
        num_inputs: Number of input channels (default: 2)

    Attributes:
        gains (List[float]): List of gain values for each input channel (default: 1.0)

    Example:
        >>> mixer = Mixer(num_inputs=4)
        >>> mixer.set_parameter("gain1", 0.8)  # Set channel 1 gain
        >>> mixer.set_parameter("gain2", 0.6)  # Set channel 2 gain
        >>> mixed_signal = mixer.process(input_signals)[0]
    """

    def __init__(self, num_inputs: int = 2):
        super().__init__(input_count=num_inputs, output_count=1)
        self.gains = [1.0] * num_inputs

    def set_parameter(self, name: str, value: ParameterType):
        """
        Set mixer parameters.

        Args:
            name: Parameter name. Supported parameters:
                - "gain1", "gain2", ..., "gainN": Gain for input channel N
                  where N is the channel number (1-indexed)
            value: Gain value (typically 0.0 to 1.0, but can exceed 1.0 for amplification)

        Note:
            Gain values can be negative for phase inversion effects.
            Values greater than 1.0 will amplify the signal and may cause clipping.
        """
        if name.startswith("gain") and name[4:].isdigit():
            input_idx = int(name[4:]) - 1  # gain1 -> index 0
            if 0 <= input_idx < self.input_count:
                self.gains[input_idx] = float(value)
            else:
                print(f"Warning: Gain index {input_idx+1} out of range for Mixer")

        else:
            print(f"Warning: Unknown parameter {name} for Mixer")

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        """
        Mix multiple input audio signals into a single output signal.

        Combines all input AUDIO signals by summing them with their respective
        gain values applied. Non-audio signals are ignored. If no inputs are
        provided or no audio signals are present, outputs silence (0.0).

        Args:
            inputs: List of input signals to mix. Only AUDIO type signals are processed.

        Returns:
            List containing one AUDIO signal with the mixed output

        Note:
            The output may exceed the normal [-1.0, 1.0] range if input signals
            and gains combine to produce values outside this range. Consider
            applying limiting or normalization if needed.
        """
        output_value = 0.0
        if inputs:
            for i, signal in enumerate(inputs):
                if i < len(self.gains) and signal.type == SignalType.AUDIO:
                    output_value += signal.value * self.gains[i]
        return [Signal(SignalType.AUDIO, output_value)]
