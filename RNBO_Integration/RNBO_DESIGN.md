# RNBO Integration Design

This document outlines the architecture and design decisions for integrating Cycling '74 RNBO C++ exported patches into the Signals framework.

## 1. Integration Flow Overview
The integration connects the RNBO exported C++ patch natively into the Rust-based modular synthesis graph.
The data flows as follows:

`RNBO (Max/MSP patch)` → `C++ Export` → `C++ MinimalEngine Wrapper (rnbo_bridge.cpp)` → `cxx Rust Bindings` → `Rust ModuleGraph`

1. **C++ Export**: RNBO code is exported as `analogosc.cpp.h` (and associated common headers).
2. **C++ Wrapper (`rnbo_bridge.cpp`)**: A minimal C++ layer that uses `RNBO::MinimalEngine<>` and a custom memory allocator (`tlsf`). This avoids heavy OS/audio dependencies while managing the RNBO state.
3. **Rust Bindings (`cxx`)**: The `cxx` crate securely bridges the C++ `RnboHost` object into Rust.
4. **Rust Module (`RNBOModule`)**: Inherits from the base `Module` trait in Rust, processing inputs and outputs using block-based (buffer) arrays directly.

## 2. Audio Processing Pipeline & The Role of Python
Signals employs a strict separation of concerns regarding languages.

- **Rust Core Engine (`rust/`)**: All real-time DSP, graph routing, and block processing occur strictly in Rust. The RNBO object executes rapidly via the C++ to Rust bridge.
- **Python Framework (`python/`)**: Python's role is strictly limited to the visual node editor UI, high-level API usage, test orchestration, and patch file generation.
- **Strict Boundary**: There are absolutely no cross-language bindings (e.g., `pyo3`, `pybind11`) passing real-time audio data between Python and Rust. Python does not interfere with the real-time audio DSP pipeline.

## 3. History: Migration to Rust
Originally, Signals used a Python core engine with sample-by-sample processing, and RNBO was bound using `pybind11`. This architecture suffered from GIL overhead and performance bottlenecks.

The framework has now fully transitioned to the Rust Core Engine.
- The `pybind11` extension and Python `RNBOOscillator` module have been discarded.
- The raw RNBO C++ export (`analogosc.cpp.h` and common headers) are now the primary DSP components.
- The engine uses block-based buffer arrays natively in Rust for high-performance audio synthesis.
