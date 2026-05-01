import numpy as np

from ..core.module import Module, Signal, SignalType
from ..core.context import get_sample_rate_or_default

try:
    from .. import rnbo_osc
except ImportError:
    rnbo_osc = None


class RNBOOscillator(Module):
    """
    Oscillator wrapper for RNBO exported code.
    This module uses block processing under the hood.
    """

    def __init__(self, sample_rate: int | None = None, mode: int = 2):
        super().__init__(input_count=1, output_count=1)
        self.sample_rate = sample_rate or get_sample_rate_or_default()
        self.mode = mode

        if rnbo_osc is None:
            raise ImportError("rnbo_osc extension not found. Make sure it is built.")

        self._rnbo_osc = rnbo_osc.RNBOOscillator()
        self._rnbo_osc.set_sample_rate(self.sample_rate)

        # In this RNBO patch:
        # mode parameter index is 0.
        # values: 0=noise, 1=sine, 2=saw, 3=triangle, 4=square, 5=pulse
        # we need to normalize to [0, 1] as RNBO parameter API in C++ wrapper takes normalized?
        # Actually in RNBO export C++ setParameterValueNormalized sets normalized. Let's use setParameterValue.

        self.set_parameter("mode", mode)

    def set_parameter(self, name: str, value: float | int | str | bool):
        if name == "mode":
            self.mode = int(value)
            # The description.json says steps=6, min=0, max=5.
            # setParameterValueNormalized would map 0-5 to 0.0-1.0.
            normalized_value = self.mode / 5.0
            self._rnbo_osc.set_parameter(0, normalized_value)
        else:
            super().set_parameter(name, value)

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        # Support block processing

        if inputs is None or len(inputs) == 0:
            # Generate a block of silence/0Hz? Actually if no input, it just outputs 0 frequency.
            # But the user might want to process sample by sample for now until we fully migrate to block processing.
            # Let's handle both.
            block_size = 1
            freq = np.zeros(block_size)
        else:
            input_signal = inputs[0]
            if isinstance(input_signal.value, np.ndarray):
                freq = input_signal.value
                block_size = len(freq)
            else:
                freq = np.array([input_signal.value])
                block_size = 1

        output_data = self._rnbo_osc.process(freq)

        if block_size == 1:
            return [Signal(SignalType.AUDIO, output_data[0])]
        else:
            return [Signal(SignalType.AUDIO, output_data)]
