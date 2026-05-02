import pytest
import numpy as np

from signals.modules.rust_oscillator import RustOscillator
from signals.core.module import Signal, SignalType

@pytest.mark.unit
def test_rust_oscillator_sine_output():
    osc = RustOscillator(sample_rate=48000, mode=1) # Sine
    freq_sig = Signal(SignalType.AUDIO, np.array([480.0] * 64, dtype=np.float32))

    outputs = osc.process([freq_sig])

    assert len(outputs) == 1
    assert outputs[0].type == SignalType.AUDIO
    out_arr = outputs[0].value

    assert isinstance(out_arr, np.ndarray)
    assert len(out_arr) == 64

    # Very basic check: sine wave starts at 0, goes up
    # 480 Hz at 48000 Hz = 100 samples per cycle
    assert abs(out_arr[0] - 0.0) < 1e-5
    assert out_arr[1] > 0.0
