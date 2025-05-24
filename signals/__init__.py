"""
Signals package for audio synthesis and signal processing.

This package provides a modular synthesizer framework with the following components:

- Module: Base class for all signal processing modules
- Signal: Container for audio, control, trigger, and frequency signals
- Oscillator: Various waveform generators (sine, square, saw, triangle, noise)
- EnvelopeADSR: Attack-Decay-Sustain-Release envelope generator
- Mixer: Multi-input audio mixer with per-channel gain control
- OutputWav: WAV file output module
- DSP utilities: Low-level audio processing functions

Example:
    Basic synthesis setup:
    
    >>> from signals import Oscillator, EnvelopeADSR, OutputWav
    >>> osc = Oscillator(sample_rate=48000)
    >>> env = EnvelopeADSR(sample_rate=48000)
    >>> output = OutputWav("output.wav", sample_rate=48000)
"""

from .module import Module, Signal, ParameterType, SignalType
from .oscillator import Oscillator
from .envelope import EnvelopeADSR
from .mixer import Mixer
from .output import OutputWav
from .dsp import write_wav
