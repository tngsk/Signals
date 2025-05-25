# Signals - Modular Audio Synthesizer Framework

A Python-based modular synthesizer framework for audio synthesis, signal processing, and music production.

## Overview

Signals provides a comprehensive toolkit for building digital audio synthesizers with a focus on modularity, performance, and ease of use. The framework supports both programmatic API usage and declarative patch configurations through YAML files.

## Features

### Core Capabilities
- **Modular Architecture**: Pluggable modules for oscillators, envelopes, mixers, and effects
- **Context-Based Management**: Automatic sample rate consistency across synthesis graph
- **Signal Processing**: High-performance audio processing with 32-bit float precision
- **Patch System**: YAML-based configuration for complex synthesizer setups
- **Template Engine**: Parameterized patches with Jinja2 templating
- **Professional Logging**: Comprehensive logging system for debugging and monitoring

### Audio Modules
- **Oscillators**: Sine, square, triangle, sawtooth, and noise generators
- **Envelopes**: ADSR envelope generators with flexible timing
- **Mixers**: Multi-input audio mixing with individual gain controls
- **VCA**: Voltage-controlled amplifiers for modulation
- **Output**: WAV file export and real-time audio processing

### Development Tools
- **Performance Testing**: Automated performance benchmarking and regression detection
- **Memory Profiling**: Memory leak detection and usage monitoring
- **Debug Tools**: Detailed profiling and execution analysis
- **Comprehensive Testing**: Unit, integration, and performance test suites

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/signals.git
cd signals

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Context-Based Sample Rate Management

Signals now supports automatic sample rate management through context managers, eliminating the need to specify sample rates for every module:

```python
from signals import synthesis_context, SynthEngine, Oscillator, EnvelopeADSR, VCA

# Method 1: Using synthesis_context function
with synthesis_context(sample_rate=48000):
    osc = Oscillator()        # Automatically uses 48kHz
    env = EnvelopeADSR()      # Automatically uses 48kHz
    vca = VCA()               # Automatically uses 48kHz

# Method 2: Using SynthEngine as context manager
with SynthEngine(sample_rate=44100) as engine:
    osc = Oscillator()        # Automatically uses 44.1kHz
    env = EnvelopeADSR()      # Automatically uses 44.1kHz

# Method 3: Traditional explicit approach (still supported)
osc = Oscillator(sample_rate=48000)
env = EnvelopeADSR(sample_rate=48000)
```

**Benefits:**
- 🎯 **Simplified Code**: No need to pass sample rate to every module
- 🔒 **Guaranteed Consistency**: All modules automatically use the same sample rate
- 🔄 **Easy Quality Switching**: Change sample rate in one place
- 🔙 **Backward Compatible**: Explicit sample rates still work
- 🏗️ **Clean Factories**: Create module sets without repetitive parameters

**Different Quality Levels:**
```python
# High-quality synthesis (96kHz)
with synthesis_context(sample_rate=96000):
    hq_synth = create_melodic_synth()

# Standard quality (48kHz)
with synthesis_context(sample_rate=48000):
    std_synth = create_drum_synth()

# Demo quality (22.05kHz)
with synthesis_context(sample_rate=22050):
    demo_synth = create_bass_synth()
```

### Basic Usage

```python
from signals import SynthEngine, write_wav

# Create and use synthesis engine as context manager
with SynthEngine(sample_rate=48000) as engine:
    # Load a patch configuration
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    # Render audio
    audio = engine.render(duration=2.0)
    
    # Save to file
    write_wav("output.wav", audio, sample_rate=48000)
```

### Programmatic Synthesis

```python
from signals import synthesis_context, Oscillator, EnvelopeADSR, VCA, Mixer

# Use context-based sample rate management
with synthesis_context(sample_rate=48000):
    # Create modules without explicit sample rate
    osc = Oscillator()
    osc.set_parameter("frequency", 440.0)
    osc.set_parameter("waveform", "sine")

    env = EnvelopeADSR()
    env.set_parameter("attack", 0.1)
    env.set_parameter("decay", 0.2)
    env.set_parameter("sustain", 0.7)
    env.set_parameter("release", 0.5)

    vca = VCA()

    # Process audio
    env.trigger_on()
    for _ in range(1000):
        osc_signal = osc.process()[0]
        env_signal = env.process()[0]
        output = vca.process([osc_signal, env_signal])[0]
        print(output.value)
```

## Project Structure

```
Signals/
├── src/signals/              # Main package
│   ├── core/                 # Base classes and utilities
│   ├── modules/              # Audio processing modules
│   └── processing/           # High-level engines
├── tests/                    # Comprehensive test suite
│   ├── test_modules.py       # Module unit tests
│   ├── test_engine.py        # Engine integration tests
│   ├── test_performance.py   # Performance benchmarks
│   └── test_debugging.py     # Profiling and analysis
├── scripts/                  # Demo and tutorial scripts
├── examples/                 # Sample patches and templates
└── temp/                     # Generated audio and debug files
```

## Running Examples

### Interactive Demos

```bash
# Context-based sample rate management demo
uv run python scripts/demo_context_usage.py

# Guided interactive demo
uv run python scripts/demo_phase2_guided.py

# Basic synthesis demo
uv run python scripts/demo_phase2.py

# Logging system demonstration
uv run python scripts/demo_logging.py

# Interactive tutorial
uv run python scripts/tutorial_phase2.py
```

### Single Patch Rendering

```bash
# Render a patch file to audio
uv run python render_patch.py examples/patches/basic_synth.yaml

# With custom output filename
uv run python render_patch.py examples/patches/basic_synth.yaml --output my_audio.wav
```

## Patch Configuration

Patches can be loaded and rendered using context-based sample rate management:

```python
from signals import SynthEngine

# Engine manages sample rate context automatically
with SynthEngine(sample_rate=48000) as engine:
    # All modules in the patch use the engine's sample rate
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    audio = engine.render(duration=2.0)
```

### Basic Patch Format

```yaml
name: "Basic Synthesizer"
description: "Simple oscillator with envelope"

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: 440.0
      waveform: "sine"
      amplitude: 0.8

  env1:
    type: "envelope_adsr"
    parameters:
      attack: 0.02
      decay: 0.1
      sustain: 0.7
      release: 0.2

  vca1:
    type: "vca"
    parameters:
      gain: 1.0

connections:
  - from: "osc1.0"
    to: "vca1.0"
  - from: "env1.0"
    to: "vca1.1"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 1.0
    action: "release"
    target: "env1"
```

**Note**: With context-based management, patches automatically inherit the engine's sample rate, ensuring all modules are consistent without manual configuration.

### Template System

```yaml
name: "{{ patch_name | default('Parametric Synth') }}"

variables:
  base_freq: 440.0
  env_attack: 0.02

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ base_freq * frequency_ratio | default(1.0) }}
      waveform: "{{ waveform | default('sine') }}"
      amplitude: {{ amplitude | default(0.8) }}

  env1:
    type: "envelope_adsr"
    parameters:
      attack: {{ env_attack }}
      decay: {{ decay_time | default(0.1) }}
      sustain: {{ sustain_level | default(0.7) }}
      release: {{ release_time | default(0.2) }}
```

## Performance Characteristics

### Real-time Requirements
- **Module Processing**: < 20μs per sample (48kHz)
- **Audio Rendering**: < 5x realtime factor for typical patches
- **Memory Usage**: < 50MB for normal operation
- **Latency**: Suitable for real-time audio applications

### Supported Configurations
- **Sample Rates**: 44.1kHz, 48kHz, 96kHz
- **Bit Depths**: 16-bit, 24-bit, 32-bit float
- **Channels**: Mono, stereo (expandable)
- **Buffer Sizes**: 64-4096 samples

## Architecture

### Design Principles

1. **Modularity**: Independent, composable audio processing units
2. **Context-Aware**: Automatic sample rate consistency without explicit configuration
3. **Performance**: Optimized for real-time audio processing
4. **Flexibility**: Support for both programmatic and declarative usage
5. **Quality**: Comprehensive testing and continuous integration
6. **Maintainability**: Clean code with professional development practices

### Signal Flow

```
Input → Module → Signal → Module → ... → Output
  ↑        ↓        ↑        ↓           ↓
Parameters  Process   Route   Process    WAV/Stream
```

### Module Types

- **Generators**: Create audio signals (oscillators, noise)
- **Processors**: Modify audio signals (filters, effects)
- **Controllers**: Generate control signals (envelopes, LFOs)
- **Utilities**: Mix, route, and output signals (mixers, output)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by modular synthesizer design principles
- Built with modern Python development practices
- Comprehensive testing inspired by professional audio software development
