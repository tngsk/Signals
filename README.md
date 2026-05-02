# Signals - Modular Audio Synthesizer Framework

A modern modular synthesizer framework featuring a high-performance Rust core engine, a Python integration framework, and a visual Node Editor.

## Overview

Signals provides a toolkit for building digital audio synthesizers, focusing on modularity, performance, and flexibility. The architecture consists of:

1. **Rust Core Engine (`rust/`)**: Block-based (buffer) audio processing module graph and C++ RNBO patches natively bound via `cxx`.
2. **Python Framework (`python/`)**: A high-level API and declarative YAML patch configurations that bridge down to the Rust engine.
3. **Visual Node Editor (`python/scripts/node_editor/`)**: A frontend for designing synthesizer patches visually in Vanilla JS + HTML5, interacting with a Python backend.

## Project Structure

```
Signals/
├── python/              # Python framework, API bindings, scripts, node editor
│   ├── src/signals/     # Main Python package
│   ├── tests/           # Python test suite
│   ├── scripts/         # Node editor and standalone CLI tools
│   └── examples/        # Example patches and configurations
├── rust/                # Core DSP engine in Rust
│   ├── src/             # Core Rust modules
│   └── rnbo/            # RNBO C++ bridge and TLSF integration
├── audio/               # Directory for audio file input/output
└── RNBO_Integration/    # Resources related to RNBO patch generation
```

## Getting Started

### Prerequisites

- **Python**: 3.12+ (managed via `uv`)
- **Rust**: Latest stable toolchain (via `rustup`)
- **CMake / C++ Compiler**: Required for RNBO C++ bridging

### 1. Rust Core Engine

The core audio engine is written in Rust.

```bash
cd rust
cargo build --release
cargo test
```

### 2. Python Framework

The Python framework uses `uv` for dependency management.

```bash
cd python
uv sync
uv run pytest
```

### 3. Visual Node Editor

To start the visual node editor backend and serve the Vanilla JS UI:

```bash
cd python
uv run python scripts/node_editor/server.py
```
Open a browser at `http://localhost:8000` to interact with the Node Editor.

### 4. CLI Tools

Standalone CLI tools for testing individual modules are available. The framework generates audio files in the top-level `audio/` folder.

```bash
cd python
uv run signals-osc --frequency 440.0 --waveform sine
uv run signals-filter --cutoff 1000.0 --resonance 0.5
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
