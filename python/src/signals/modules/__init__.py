"""
Signal processing modules for the Signals synthesizer framework.

This module contains all the synthesizer modules that generate, process,
or output audio signals:
- Oscillator: Various waveform generators
- EnvelopeADSR: Attack-Decay-Sustain-Release envelope generator
- Mixer: Multi-input audio mixer
- OutputWav: WAV file output module
"""

from .envelope import EnvelopeADSR
from .filter import Filter
from .lfo import LFO
from .mixer import Mixer
from .oscillator import Oscillator
from .output import OutputWav
from .vca import VCA

__all__ = [
    "Oscillator",
    "EnvelopeADSR",
    "Mixer",
    "OutputWav",
    "VCA",
    "Filter",
    "LFO"
]
