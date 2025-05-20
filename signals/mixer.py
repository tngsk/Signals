from .module import Module, Signal, SignalType, ParameterType
import numpy as np

class Mixer(Module):
    def __init__(self, num_inputs: int = 2):
        super().__init__(input_count=num_inputs, output_count=1)
        self.gains = [1.0] * num_inputs

    def set_parameter(self, name: str, value: ParameterType):
        if name.startswith("gain") and name[4:].isdigit():
            input_idx = int(name[4:]) -1 # gain1 -> index 0
            if 0 <= input_idx < self.input_count:
                self.gains[input_idx] = float(value)
            else:
                print(f"Warning: Gain index {input_idx+1} out of range for Mixer")
        else:
            print(f"Warning: Unknown parameter {name} for Mixer")

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        output_value = 0.0
        if inputs:
            for i, signal in enumerate(inputs):
                if i < len(self.gains) and signal.type == SignalType.AUDIO:
                    output_value += signal.value * self.gains[i]
        return [Signal(SignalType.AUDIO, output_value)]
