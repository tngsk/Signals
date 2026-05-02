# RNBO Integration Report & Rust Transition Strategy

## Overview

The Signals engine is transitioning away from C++ RNBO patches wrapped via `pybind11` and the Minimal Export layer, moving towards native, pure Rust implementations for its core synthesis and DSP modules.

## Current State

Historically, the project utilized RNBO's C++ export along with a `MinimalEngine` and custom memory allocation (`tlsf`) to bridge complex signal processing networks into the Python core. Modules such as `RNBOOscillator` relied on an external C++ `.so` to generate waveforms.

## Transition to Pure Rust

A new phase is now in progress where these modules are being systematically rewritten in pure Rust:
- The new `rnbo_analog_rust` crate introduces a `AnalogOscillator` capable of producing the same standard waveforms (noise, sine, saw, triangle, square, pulse).
- Python integration is achieved seamlessly through PyO3, which offers high-performance bindings without the overhead of C++ toolchains.
- The wrapper class `RustOscillator` mirrors the interface of the existing `RNBOOscillator`, making it a drop-in replacement within the Signals ecosystem.

## Future Development Flow

For new DSP components, the recommended approach is:
1. **Prototype in RNBO/Max** to verify the DSP logic aurally and mathematically.
2. **Export to C++ (Optional)** for reference and testing using the legacy Minimal Engine if necessary.
3. **Implement in Rust** within a crate (e.g., `rnbo_analog_rust` or `signals_core`), utilizing the PyO3 bridge to expose it to the Python wrapper layer.
4. **Gradually Deprecate** the `pybind11` integration. Once all standard modules are fully supported in Rust, the C++ build requirement will become entirely optional or fully removed.
