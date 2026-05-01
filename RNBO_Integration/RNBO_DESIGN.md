# RNBO Integration Design (Phase 1 PoC)

This document outlines the architecture and design decisions for integrating Cycling '74 RNBO C++ exported patches into the Signals framework.

## 1. Integration Flow Overview
The integration connects the RNBO exported C++ patch directly into the Python-based modular synthesis graph.
The data flows as follows:

`RNBO (Max/MSP patch)` → `C++ Export` → `pybind11 C++ Wrapper` → `RNBOOscillator (Python Module)` → `Signals Graph`

1. **C++ Export**: RNBO code is exported as `analogosc.cpp.h` (and associated common headers).
2. **pybind11 Wrapper (`src/signals/rnbo_osc.cpp`)**: We compile this into a Python extension. It instantiates the `RNBO::rnbomatic<>` patcher, exposes parameter updates, and processes audio by ferrying data via NumPy arrays.
3. **Python Module (`src/signals/modules/rnbo_oscillator.py`)**: Inherits from the base `Module` and hides the C++ interaction from the rest of the application, seamlessly translating Signals' event/routing system to the underlying RNBO object.

## 2. Policy: Sample-by-Sample vs. Block Processing
Signals was originally designed around sample-by-sample processing (`process(self, inputs)` processing length 1 data). However, RNBO is optimized for block processing.

Our policy for the PoC bridging phase is **Hybrid Compatibility**:
- The C++ extension accepts numpy arrays of any block size dynamically.
- `RNBOOscillator` inspects the input:
  - If a scalar float (or size-1 array) is provided, it passes an array of size 1 to C++ and unpacks the returned array into a single float, complying with the current graph loop.
  - If a Numpy array is provided, it processes the entire block natively and returns a Numpy array.
- This allows `RNBOOscillator` to run as a drop-in replacement today, while perfectly positioning it to receive full block arrays once the `ModuleGraph` is upgraded from its sample-by-sample loop.

## 3. Future Roadmap: Rust Migration
The current Python binding using `pybind11` introduces overhead bridging GIL, memory allocation, and Python types, especially at small block sizes.

In Phase 2, the core Engine (`ModuleGraph`, `SignalStore`, `Scheduler`) will be migrated to Rust.
At that stage:
- **What will be replaced**: The `pybind11` extension and `RNBOOscillator.py` module will be completely discarded.
- **What will remain**: The raw RNBO C++ export (`analogosc.cpp.h` and common headers).
- **How it integrates**: Rust will natively host the RNBO C++ object using a bridge (e.g., `cxx` crate) and process audio directly using safe Rust buffer arrays. The hybrid sample/block workaround will be removed, as the entire audio bus will be purely block-based using CPAL.
