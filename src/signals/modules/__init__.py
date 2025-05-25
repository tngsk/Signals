"""
Signal processing modules for the Signals synthesizer framework.

This module contains all the synthesizer modules that generate, process,
or output audio signals:
- Oscillator: Various waveform generators
- EnvelopeADSR: Attack-Decay-Sustain-Release envelope generator
- Mixer: Multi-input audio mixer
- OutputWav: WAV file output module
"""

from .oscillator import Oscillator
from .envelope import EnvelopeADSR
from .mixer import Mixer
from .output import OutputWav
from .vca import VCA
from .filter import Filter
from .lfo import LFO

__all__ = [
    "Oscillator",
    "EnvelopeADSR", 
    "Mixer",
    "OutputWav",
    "VCA",
    "Filter",
    "LFO"
]