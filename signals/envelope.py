from .module import Module, Signal, SignalType, ParameterType
import numpy as np

class EnvelopeADSR(Module):
    def __init__(self, sample_rate: int):
        super().__init__(input_count=1, output_count=1) # Input for trigger
        self.sample_rate = sample_rate
        self.attack_time: float = 0.05  # seconds
        self.decay_time: float = 0.1   # seconds
        self.sustain_level: float = 0.7 # 0.0 to 1.0
        self.release_time: float = 0.2  # seconds

        self._attack_samples: int = int(self.attack_time * self.sample_rate)
        self._decay_samples: int = int(self.decay_time * self.sample_rate)
        self._release_samples: int = int(self.release_time * self.sample_rate)

        self._phase: int = 0 # 0: idle, 1: attack, 2: decay, 3: sustain, 4: release
        self._current_sample: int = 0
        self._value: float = 0.0
        self._note_on: bool = False

    def set_parameter(self, name: str, value: ParameterType):
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
        self._note_on = True
        self._phase = 1 # Attack
        self._current_sample = 0

    def trigger_off(self):
        self._note_on = False
        if self._phase != 0: # If not idle
            self._phase = 4 # Release
            self._current_sample = 0

    def process(self, inputs: list[Signal] | None = None) -> list[Signal]:
        if inputs and inputs[0].type == SignalType.TRIGGER and inputs[0].value > 0.5 and not self._note_on:
            self.trigger_on()
        elif inputs and inputs[0].type == SignalType.TRIGGER and inputs[0].value <= 0.5 and self._note_on:
            self.trigger_off()

        if self._phase == 1: # Attack
            self._value = self._current_sample / self._attack_samples if self._attack_samples > 0 else 1.0
            if self._current_sample >= self._attack_samples:
                self._phase = 2 # Decay
                self._current_sample = 0
        elif self._phase == 2: # Decay
            progress = self._current_sample / self._decay_samples if self._decay_samples > 0 else 1.0
            self._value = 1.0 - (1.0 - self.sustain_level) * progress
            if self._current_sample >= self._decay_samples:
                self._phase = 3 # Sustain
        elif self._phase == 3: # Sustain
            self._value = self.sustain_level
            if not self._note_on: # Should have been caught by trigger_off, but as a safeguard
                self._phase = 4 # Release
                self._current_sample = 0
        elif self._phase == 4: # Release
            self._value = self.sustain_level * (1.0 - (self._current_sample / self._release_samples if self._release_samples > 0 else 1.0))
            if self._current_sample >= self._release_samples:
                self._phase = 0 # Idle
                self._value = 0.0

        if self._phase != 0:
            self._current_sample += 1

        return [Signal(SignalType.CONTROL, np.clip(self._value, 0.0, 1.0))]
