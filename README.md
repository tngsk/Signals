# Signals: Modular Synthesizer Engine

A high-performance, modular synthesizer engine implemented in Python, designed for audio synthesis, signal processing, and external parameter exploration. Signals provides a flexible framework for creating and processing audio through interconnected modules with YAML-based patch configuration and Jinja2 templating support.

## Features

- **Modular Architecture**: Core/Modules/Processing hierarchy with clean separation of concerns
- **YAML Patch System**: Load synthesizer configurations from YAML files with validation
- **Template System**: Jinja2-powered parameterized patches for systematic exploration
- **High Performance**: Optimized module graph execution (2300x speedup achieved)
- **External Control**: Programmatic API for integration with parameter exploration tools
- **Audio Feature Extraction**: Automatic analysis of generated audio
- **Batch Processing**: Generate multiple variations from templates

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Signals

# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv sync --extra dev

# Or install with pip
pip install -e .
```

### Basic Usage

```python
from signals import SynthEngine

# Create engine
engine = SynthEngine(sample_rate=48000)

# Load and render a patch
patch = engine.load_patch("examples/patches/basic_synth.yaml")
audio_data = engine.render(duration=2.0)

# Extract features
features = engine.export_features(audio_data)
print(f"Peak: {features['peak']:.3f}, RMS: {features['rms']:.3f}")
```

## Command Line Usage

### Running Demos

```bash
# Interactive guided demo (recommended)
uv run python scripts/demo_phase2_guided.py

# Basic demo
uv run python scripts/demo_phase2.py

# Interactive tutorial
uv run python scripts/tutorial_phase2.py

# Legacy Phase 1 demo
uv run python main.py --silence 0.5
```

### Quick Audio Generation

```bash
# Generate audio from basic patch
uv run python -c "
from signals import SynthEngine
engine = SynthEngine(sample_rate=48000)
patch = engine.load_patch('examples/patches/basic_synth.yaml')
audio = engine.render(duration=2.0)
print(f'Generated {len(audio)} samples')
"

# Generate with template variables
uv run python -c "
from signals import SynthEngine
engine = SynthEngine()
patch = engine.load_patch('examples/templates/parametric_synth.yaml', 
                         variables={'osc_freq': 880, 'env_attack': 0.05})
audio = engine.render(duration=1.0)
features = engine.export_features(audio)
print(f'Peak: {features[\"peak\"]:.3f}, RMS: {features[\"rms\"]:.3f}')
"
```

### Performance Testing

```bash
# Module performance analysis
uv run python scripts/debug_module_performance.py

# Complex patch debugging
uv run python scripts/debug_complex_patch.py

# Detailed graph profiling
uv run python scripts/debug_graph_detailed.py
```

### Running Tests

```bash
# All tests
uv run pytest

# Specific test suites
uv run pytest tests/test_basic.py      # Phase 1 tests
uv run pytest tests/test_phase2.py    # Phase 2 tests

# With coverage
uv run pytest --cov=signals
```

## Patch Configuration

### Basic Patch Format

Create a YAML file describing your synthesizer setup:

```yaml
# examples/patches/basic_synth.yaml
name: "Basic Synthesizer"
description: "Simple oscillator with ADSR envelope"
sample_rate: 48000

modules:
  osc1:
    type: "oscillator"
    parameters:
      waveform: "sine"
      frequency: 440.0
      amplitude: 0.8
      
  env1:
    type: "envelope_adsr"
    parameters:
      attack: 0.02
      decay: 0.8
      sustain: 0.0
      release: 0.1
      
  output:
    type: "output_wav"
    parameters:
      filename: "basic_synth_output.wav"

connections:
  - from: "osc1.0"
    to: "env1.0"
  - from: "env1.0" 
    to: "output.0"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 1.5
    action: "release"
    target: "env1"
```

### Parameterized Templates

Create templates with variables for parameter exploration:

```yaml
# examples/templates/parametric_synth.yaml
name: "Parametric Synthesizer"
description: "Template with variable parameters"

variables:
  osc_freq: 440.0
  osc_waveform: "sine"
  env_attack: 0.02

modules:
  osc1:
    type: "oscillator"
    parameters:
      waveform: "{{ osc_waveform | default('sine') }}"
      frequency: {{ osc_freq | default(440.0) }}
      
  env1:
    type: "envelope_adsr"
    parameters:
      attack: {{ env_attack | default(0.02) }}
```

## Programmatic API

### SynthEngine

```python
from signals import SynthEngine, PatchTemplate

# Initialize engine
engine = SynthEngine(sample_rate=48000)

# Load basic patch
patch = engine.load_patch("examples/patches/basic_synth.yaml")
audio = engine.render(duration=2.0)

# Load template with variables
template_patch = engine.load_patch(
    "examples/templates/parametric_synth.yaml",
    variables={"osc_freq": 880.0, "env_attack": 0.05}
)
audio = engine.render(duration=1.0)

# Dynamic parameter control
engine.set_module_parameter("osc1", "frequency", 660.0)
audio = engine.render(duration=0.5)

# Feature extraction
features = engine.export_features(audio)
print(f"Audio features: {features}")
```

### Template System

```python
from signals import PatchTemplate

# Create template
template = PatchTemplate("examples/templates/parametric_synth.yaml")

# Inspect variables
print(f"Variables: {template.variables}")
schema = template.get_variable_schema()
print(f"Default values: {schema}")

# Generate variations
variations = [
    {"osc_freq": 220.0, "osc_waveform": "sine"},
    {"osc_freq": 440.0, "osc_waveform": "square"},
    {"osc_freq": 880.0, "osc_waveform": "triangle"}
]

for params in variations:
    patch = template.instantiate(params)
    engine = SynthEngine()
    audio = engine.render_patch(patch, duration=1.0)
```

### Batch Processing

```python
from signals import SynthEngine

engine = SynthEngine(sample_rate=48000)

# Parameter exploration
parameter_sets = [
    {"osc_freq": 220.0, "env_attack": 0.01},
    {"osc_freq": 440.0, "env_attack": 0.05},
    {"osc_freq": 880.0, "env_attack": 0.1}
]

results = []
for i, params in enumerate(parameter_sets):
    # Load template with parameters
    patch = engine.load_patch(
        "examples/templates/parametric_synth.yaml", 
        variables=params
    )
    
    # Render audio
    audio = engine.render(duration=1.0)
    
    # Extract features
    features = engine.export_features(audio)
    
    results.append({
        "parameters": params,
        "features": features,
        "audio_length": len(audio)
    })

print(f"Generated {len(results)} variations")
```

## Available Modules

### Oscillator
- **Types**: sine, square, triangle, saw, noise
- **Parameters**: frequency, waveform, amplitude
- **Inputs**: 1 (frequency modulation)
- **Outputs**: 1 (audio signal)

### ADSR Envelope
- **Parameters**: attack, decay, sustain, release
- **Inputs**: 1 (audio input)
- **Outputs**: 1 (modulated signal)
- **Methods**: trigger_on(), trigger_off()

### Mixer
- **Parameters**: num_inputs, gain1, gain2, ... (per channel)
- **Inputs**: configurable (default: 2)
- **Outputs**: 1 (mixed audio)

### WAV Output
- **Parameters**: filename
- **Inputs**: 1 (audio input)
- **Outputs**: 0 (file output)

## Project Structure

```
Signals/
├── src/signals/              # Main package
│   ├── core/                 # Base classes and utilities
│   │   ├── module.py         # Module base class, Signal types
│   │   └── dsp.py            # DSP utilities
│   ├── modules/              # Synthesizer modules
│   │   ├── oscillator.py     # Waveform generators
│   │   ├── envelope.py       # ADSR envelope
│   │   ├── mixer.py          # Audio mixer
│   │   └── output.py         # WAV file output
│   └── processing/           # High-level engines
│       ├── engine.py         # SynthEngine main API
│       ├── graph.py          # ModuleGraph execution
│       └── patch.py          # Patch loading system
├── examples/                 # Sample files
│   ├── patches/              # Basic patch files
│   └── templates/            # Parameterized templates
├── scripts/                  # Demo and debug tools
├── tests/                    # Test suite
└── temp/                     # Generated files (git-ignored)
```

## Performance

Signals achieves high-performance audio synthesis through optimized algorithms:

- **Module processing**: 0.001-0.004ms per sample
- **Graph execution**: 0.006ms per sample (2300x optimization achieved)
- **Real-time factor**: 0.38x (processes 1 second of audio in 0.38 seconds)
- **Memory efficient**: Optimized buffer management and caching

## Development

### Requirements

- Python 3.11+
- NumPy (numerical computing)
- PyYAML (patch file loading)
- Jinja2 (template system)
- SciPy (advanced DSP functions)

### Development Tools

- **uv**: Fast package management and virtual environments
- **pytest**: Testing framework with coverage
- **ruff**: Fast linting
- **black**: Code formatting

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `uv run pytest`
5. Check formatting: `uv run ruff check src/`
6. Submit a pull request

## Architecture

Signals follows a modular, layered architecture:

### Core Layer
- `Module`: Abstract base class for all signal processors
- `Signal`: Typed container for audio/control/trigger data
- `SignalType`: Enumeration of signal types

### Modules Layer
- Concrete implementations of synthesizer components
- Each module processes signals according to its parameters
- Modules can be interconnected via the graph system

### Processing Layer
- `SynthEngine`: High-level API for synthesis control
- `ModuleGraph`: Manages module connections and execution order
- `Patch`/`PatchTemplate`: Configuration loading and templating

## External Integration

Signals is designed for integration with external parameter exploration tools:

```python
# Example external program integration
import numpy as np
from signals import SynthEngine

def explore_parameter_space(template_file, param_ranges):
    engine = SynthEngine()
    results = []
    
    for freq in param_ranges['frequency']:
        for attack in param_ranges['attack']:
            params = {'osc_freq': freq, 'env_attack': attack}
            
            patch = engine.load_patch(template_file, variables=params)
            audio = engine.render(duration=1.0)
            features = engine.export_features(audio)
            
            results.append({
                'parameters': params,
                'peak': features['peak'],
                'rms': features['rms'],
                'spectral_centroid': features['spectral_centroid']
            })
    
    return results

# Run exploration
param_ranges = {
    'frequency': [220, 440, 880],
    'attack': [0.01, 0.05, 0.1]
}

results = explore_parameter_space(
    "examples/templates/parametric_synth.yaml",
    param_ranges
)
```

## Real-World Usage Examples

### Example 1: Generate Frequency Sweep

```python
from signals import SynthEngine

engine = SynthEngine()
frequencies = [220, 330, 440, 550, 660]

for freq in frequencies:
    patch = engine.load_patch(
        "examples/templates/parametric_synth.yaml",
        variables={"osc_freq": freq, "output_filename": f"sweep_{freq}hz.wav"}
    )
    audio = engine.render(duration=1.0)
    print(f"Generated {freq}Hz: {len(audio)} samples")
```

### Example 2: Waveform Comparison

```python
from signals import SynthEngine

engine = SynthEngine()
waveforms = ["sine", "square", "triangle", "saw"]

results = []
for waveform in waveforms:
    patch = engine.load_patch(
        "examples/templates/parametric_synth.yaml",
        variables={"osc_waveform": waveform}
    )
    audio = engine.render(duration=0.5)
    features = engine.export_features(audio)
    
    results.append({
        "waveform": waveform,
        "peak": features["peak"],
        "rms": features["rms"]
    })
    
for result in results:
    print(f"{result['waveform']:8s}: peak={result['peak']:.3f}")
```

### Example 3: Envelope Analysis

```python
from signals import SynthEngine

engine = SynthEngine()
attack_times = [0.01, 0.05, 0.1, 0.2]

for attack in attack_times:
    patch = engine.load_patch(
        "examples/templates/parametric_synth.yaml",
        variables={"env_attack": attack}
    )
    audio = engine.render(duration=1.0)
    features = engine.export_features(audio)
    
    print(f"Attack {attack:4.2f}s: RMS={features['rms']:.3f}")
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, ensure the package is installed
uv sync --extra dev

# Or use development mode
pip install -e .
```

#### Audio Generation Issues
```python
# If audio files are not generated, check temp directory
import os
os.makedirs("temp/audio", exist_ok=True)

# Verify patch file syntax
from signals import Patch
try:
    patch = Patch.from_file("your_patch.yaml")
    print("Patch is valid")
except Exception as e:
    print(f"Patch error: {e}")
```

#### Performance Issues
```bash
# Run performance diagnostics
uv run python scripts/debug_module_performance.py

# Check for infinite loops in patches
uv run python scripts/debug_complex_patch.py
```

#### Template Variable Errors
```python
# Check available variables in template
from signals import PatchTemplate
template = PatchTemplate("your_template.yaml")
print(f"Available variables: {template.variables}")
print(f"Default values: {template.get_variable_schema()}")
```

### Known Limitations

- WAV output files are created in the project directory unless specified otherwise
- Some spectral features may produce warnings with certain audio configurations
- Template variables must be properly defined with default values using Jinja2 syntax
- Module connections are validated at patch load time, not at runtime

### Getting Help

1. Check the examples in `examples/` directory
2. Run the guided demo: `uv run python scripts/demo_phase2_guided.py`
3. Review test files in `tests/` for usage patterns
4. Use the interactive tutorial: `uv run python scripts/tutorial_phase2.py`

## License

[Add your license information here]

## Acknowledgments

Built with modern Python tooling and optimized for research and creative applications in audio synthesis and parameter exploration.