import pytest
import numpy as np

from signals.core.module import Signal, SignalType
from signals.modules.rnbo_oscillator import RNBOOscillator


def test_rnbo_oscillator_initialization():
    osc = RNBOOscillator(sample_rate=48000, mode=1)
    assert osc.sample_rate == 48000
    assert osc.mode == 1
    assert osc.input_count == 1
    assert osc.output_count == 1

def test_rnbo_oscillator_block_processing():
    osc = RNBOOscillator(sample_rate=48000, mode=2)
    # Give a block of size 64
    freq_block = np.full(64, 440.0)

    out_signals = osc.process([Signal(SignalType.FREQUENCY, freq_block)])
    assert len(out_signals) == 1
    out_sig = out_signals[0]

    assert out_sig.type == SignalType.AUDIO
    assert isinstance(out_sig.value, np.ndarray)
    assert len(out_sig.value) == 64

    # Check that it produces non-zero audio after some samples
    assert np.any(np.abs(out_sig.value) > 0.0)

def test_rnbo_oscillator_single_sample_processing():
    osc = RNBOOscillator(sample_rate=48000, mode=2)

    out_signals = osc.process([Signal(SignalType.FREQUENCY, 440.0)])
    assert len(out_signals) == 1
    out_sig = out_signals[0]

    assert out_sig.type == SignalType.AUDIO
    assert isinstance(out_sig.value, float)

def test_rnbo_oscillator_no_input():
    osc = RNBOOscillator()

    out_signals = osc.process()
    assert len(out_signals) == 1
    out_sig = out_signals[0]

    assert out_sig.type == SignalType.AUDIO
    assert isinstance(out_sig.value, float)

def test_rnbo_oscillator_set_parameter():
    osc = RNBOOscillator()
    osc.set_parameter("mode", 3)
    assert osc.mode == 3
