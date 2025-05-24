"""
High-level processing components for the Signals synthesizer framework.

This module contains the sophisticated processing engines that manage
the synthesis pipeline:
- SynthEngine: Main synthesis control interface
- ModuleGraph: Signal routing and execution engine
- Patch: Configuration container and loader
- PatchTemplate: Parameterized patch system
"""

from .engine import SynthEngine
from .graph import ModuleGraph
from .patch import Patch, PatchTemplate

__all__ = [
    "SynthEngine",
    "ModuleGraph",
    "Patch",
    "PatchTemplate"
]