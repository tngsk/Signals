import numpy as np

from signals.core.module import Module, Signal, SignalType
from signals.core.context import get_sample_rate_or_default

try:
    import rnbo_analog_rust
except ImportError:
    rnbo_analog_rust = None


class RustOscillator(Module):
    """
    Oscillator wrapper for the pure Rust implementation (rnbo_analog_rust).
    This module is highly compatible with the RNBOOscillator.
    """

    def __init__(self, sample_rate: int | None = None, mode: int = 2):
        super().__init__(input_count=1, output_count=1)
        self.sample_rate = sample_rate or get_sample_rate_or_default()
        self.mode = mode

        if rnbo_analog_rust is None:
            raise ImportError("rnbo_analog_rust extension not found. Make sure to build the pure Rust oscillator.")

        self._rust_osc = rnbo_analog_rust.AnalogOscillator(self.sample_rate, self.mode)

    def set_parameter(self, name: str, value: float | int | str | bool):
        if name == "mode":
            self.mode = int(value)
            self._rust_osc.set_mode(self.mode)
        elif name == "sample_rate":
            self.sample_rate = float(value)
            self._rust_osc.set_sample_rate(self.sample_rate)
        else:
            super().set_parameter(name, value)

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        if inputs is None or len(inputs) == 0:
            block_size = 1
            freq = [0.0]
        else:
            input_signal = inputs[0]
            if isinstance(input_signal.value, np.ndarray):
                # The Rust wrapper expects a list or sequence, we can convert numpy array to list
                freq = input_signal.value.tolist()
                block_size = len(freq)
            else:
                freq = [float(input_signal.value)]
                block_size = 1

        # The rust process returns a standard python list
        output_data = self._rust_osc.process(freq)

        if block_size == 1:
            return [Signal(SignalType.AUDIO, output_data[0])]
        else:
            # We return it as a numpy array for downstream compatibility
            return [Signal(SignalType.AUDIO, np.array(output_data, dtype=np.float32))]
