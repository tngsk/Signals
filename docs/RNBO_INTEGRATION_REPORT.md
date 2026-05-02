# RNBO Integration Report & Rust Transition Strategy

## Overview

The primary purpose of the Signals project is to load exported C++ patches from RNBO as modules and combine them for audio processing.

Previously, there was a misunderstanding in the documentation that the project was moving away from RNBO C++ patches towards pure Rust DSP implementations. This is incorrect. The actual transition involves replacing the legacy Python implementation (which previously handled the module graph and engine logic) with a high-performance Rust core engine.

RNBO patch construction and C++ export remain the core, ongoing foundation for creating DSP modules in this project.

## Current State

The project utilizes RNBO's C++ export along with a `MinimalEngine` and custom memory allocation (`tlsf`) to bridge complex signal processing networks.

Previously, this was integrated into a Python core. Now, these RNBO C++ patches are natively bound to the new Rust core engine via `cxx`.

## Transition to Rust Core Engine

A new phase is in progress where the core engine and module graph management are systematically rewritten in Rust:
- The Rust core engine manages the block-based (buffer) audio processing and module graph execution.
- RNBO C++ patches are bridged directly into Rust using `cxx` and the Minimal Export layer.
- Python integration is maintained as a high-level framework (e.g., for API bindings, visual node editor), but the heavy lifting of audio processing and graph execution is fully delegated to Rust.

## Future Development Flow

For new DSP components, the recommended approach is:
1. **Develop in RNBO/Max** to construct the processing patch and verify the DSP logic.
2. **Export to C++** using the RNBO export target.
3. **Load and Bind in Rust** where the RNBO C++ export is treated as a module and natively bound into the Rust engine's module graph.
4. **Combine and Process** within the Rust engine to execute the complex audio processing networks.

The Python implementation of the core engine is being deprecated in favor of this Rust architecture, while RNBO remains the absolute primary source for DSP module logic.