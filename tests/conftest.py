"""
Comprehensive test configuration and fixtures for the Signals synthesizer framework.

This file provides pytest configuration, shared fixtures, and utilities for testing
all components of the Signals framework including modules, patches, engine, and
integration scenarios.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Generator
import yaml

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import (
    SynthEngine, Oscillator, EnvelopeADSR, Mixer, VCA, OutputWav,
    Signal, SignalType, Patch, PatchTemplate, ModuleGraph,
    configure_logging, LogLevel
)


# Test Configuration
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "audio: Tests that generate or process audio"
    )
    config.addinivalue_line(
        "markers", "patch: Tests for patch loading and processing"
    )
    config.addinivalue_line(
        "markers", "logging: Tests for logging functionality"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    
    # Configure logging for tests
    configure_logging(level=LogLevel.ERROR, console=False)


@pytest.fixture(scope="session")
def sample_rates():
    """Standard sample rates for testing."""
    return [44100, 48000, 96000]


@pytest.fixture(scope="session")
def test_frequencies():
    """Standard test frequencies."""
    return [220.0, 440.0, 880.0, 1760.0]


@pytest.fixture(scope="session")
def waveform_types():
    """Available waveform types for testing."""
    return ["sine", "square", "saw", "triangle", "noise"]


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    def _generate(sample_rate=48000, duration=1.0, frequency=440.0, amplitude=0.5):
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        return amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return _generate


@pytest.fixture
def oscillator_module():
    """Create oscillator module for testing."""
    def _create(sample_rate=48000, **params):
        osc = Oscillator(sample_rate)
        for param, value in params.items():
            osc.set_parameter(param, value)
        return osc
    return _create


@pytest.fixture
def envelope_module():
    """Create envelope module for testing."""
    def _create(sample_rate=48000, **params):
        env = EnvelopeADSR(sample_rate)
        for param, value in params.items():
            env.set_parameter(param, value)
        return env
    return _create


@pytest.fixture
def vca_module():
    """Create VCA module for testing."""
    def _create(sample_rate=48000, **params):
        vca = VCA(sample_rate)
        for param, value in params.items():
            vca.set_parameter(param, value)
        return vca
    return _create


@pytest.fixture
def mixer_module():
    """Create mixer module for testing."""
    def _create(num_inputs=2, **params):
        mixer = Mixer(num_inputs)
        for param, value in params.items():
            mixer.set_parameter(param, value)
        return mixer
    return _create


@pytest.fixture
def synth_engine():
    """Create synthesis engine for testing."""
    def _create(sample_rate=48000, buffer_size=1024):
        engine = SynthEngine(sample_rate, buffer_size)
        yield engine
        engine.cleanup()
    return _create


@pytest.fixture
def basic_patch_data():
    """Basic patch data for testing."""
    return {
        'name': 'Test Patch',
        'description': 'Basic test patch',
        'sample_rate': 48000,
        'modules': {
            'osc1': {
                'type': 'oscillator',
                'parameters': {
                    'frequency': 440.0,
                    'waveform': 'sine',
                    'amplitude': 0.8
                }
            },
            'env1': {
                'type': 'envelope_adsr',
                'parameters': {
                    'attack': 0.02,
                    'decay': 0.1,
                    'sustain': 0.7,
                    'release': 0.2
                }
            },
            'vca1': {
                'type': 'vca',
                'parameters': {
                    'gain': 1.0
                }
            }
        },
        'connections': [
            {'from': 'osc1.0', 'to': 'vca1.0'},
            {'from': 'env1.0', 'to': 'vca1.1'}
        ],
        'sequence': [
            {'time': 0.0, 'action': 'trigger', 'target': 'env1'},
            {'time': 1.0, 'action': 'release', 'target': 'env1'}
        ]
    }


@pytest.fixture
def complex_patch_data():
    """Complex patch data for advanced testing."""
    return {
        'name': 'Complex Test Patch',
        'sample_rate': 48000,
        'modules': {
            'osc1': {
                'type': 'oscillator',
                'parameters': {'frequency': 440.0, 'waveform': 'sine'}
            },
            'osc2': {
                'type': 'oscillator',
                'parameters': {'frequency': 659.25, 'waveform': 'saw'}
            },
            'env1': {
                'type': 'envelope_adsr',
                'parameters': {'attack': 0.05, 'decay': 0.2, 'sustain': 0.6, 'release': 0.3}
            },
            'env2': {
                'type': 'envelope_adsr',
                'parameters': {'attack': 0.1, 'decay': 0.5, 'sustain': 0.4, 'release': 0.8}
            },
            'mixer1': {
                'type': 'mixer',
                'parameters': {'num_inputs': 2, 'gain1': 0.8, 'gain2': 0.6}
            },
            'vca1': {
                'type': 'vca',
                'parameters': {'gain': 1.0}
            },
            'output': {
                'type': 'output_wav',
                'parameters': {'filename': 'test_output.wav'}
            }
        },
        'connections': [
            {'from': 'osc1.0', 'to': 'mixer1.0'},
            {'from': 'osc2.0', 'to': 'mixer1.1'},
            {'from': 'mixer1.0', 'to': 'vca1.0'},
            {'from': 'env1.0', 'to': 'vca1.1'},
            {'from': 'vca1.0', 'to': 'output.0'}
        ],
        'sequence': [
            {'time': 0.0, 'action': 'trigger', 'target': 'env1'},
            {'time': 0.5, 'action': 'trigger', 'target': 'env2'},
            {'time': 2.0, 'action': 'release', 'target': 'env1'},
            {'time': 2.5, 'action': 'release', 'target': 'env2'}
        ]
    }


@pytest.fixture
def template_patch_content():
    """Template patch content for testing."""
    return """
name: "{{ patch_name | default('Template Patch') }}"
description: "{{ description | default('A template patch for testing') }}"
sample_rate: {{ sample_rate | default(48000) }}

variables:
  osc_freq: 440.0
  env_attack: 0.02
  env_release: 0.2

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ osc_freq | default(440.0) }}
      waveform: "{{ waveform | default('sine') }}"
      amplitude: {{ amplitude | default(0.8) }}
      
  env1:
    type: "envelope_adsr"
    parameters:
      attack: {{ env_attack | default(0.02) }}
      decay: {{ env_decay | default(0.1) }}
      sustain: {{ env_sustain | default(0.7) }}
      release: {{ env_release | default(0.2) }}
      
  vca1:
    type: "vca"
    parameters:
      gain: {{ vca_gain | default(1.0) }}

connections:
  - from: "osc1.0"
    to: "vca1.0"
  - from: "env1.0"
    to: "vca1.1"

sequence:
  - time: {{ trigger_time | default(0.0) }}
    action: "trigger"
    target: "env1"
  - time: {{ release_time | default(1.0) }}
    action: "release"
    target: "env1"
"""


@pytest.fixture
def create_patch_file():
    """Create patch file for testing."""
    def _create(temp_dir: Path, patch_data: Dict[str, Any], filename: str = "test_patch.yaml"):
        patch_file = temp_dir / filename
        with open(patch_file, 'w') as f:
            yaml.dump(patch_data, f)
        return patch_file
    return _create


@pytest.fixture
def create_template_file():
    """Create template file for testing."""
    def _create(temp_dir: Path, template_content: str, filename: str = "test_template.yaml"):
        template_file = temp_dir / filename
        template_file.write_text(template_content)
        return template_file
    return _create


@pytest.fixture
def signal_generators():
    """Signal generator functions for testing."""
    def sine_wave(frequency=440.0, amplitude=1.0, phase=0.0):
        """Generate sine wave signal."""
        def generator(sample_idx, sample_rate):
            t = sample_idx / sample_rate
            return amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return generator
    
    def square_wave(frequency=440.0, amplitude=1.0, duty_cycle=0.5):
        """Generate square wave signal."""
        def generator(sample_idx, sample_rate):
            t = sample_idx / sample_rate
            phase = (t * frequency) % 1.0
            return amplitude if phase < duty_cycle else -amplitude
        return generator
    
    def noise(amplitude=1.0, seed=None):
        """Generate noise signal."""
        if seed is not None:
            np.random.seed(seed)
        def generator(sample_idx, sample_rate):
            return amplitude * (np.random.random() * 2.0 - 1.0)
        return generator
    
    return {
        'sine': sine_wave,
        'square': square_wave,
        'noise': noise
    }


@pytest.fixture
def performance_timer():
    """Performance timing utility for tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
            return self
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
        
        def __enter__(self):
            return self.start()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
    
    return Timer


@pytest.fixture
def timeout_context():
    """Context manager for timeout-based testing."""
    import signal
    import threading
    
    class TimeoutContext:
        def __init__(self, timeout_seconds=5):
            self.timeout_seconds = timeout_seconds
            self.timed_out = False
            
        def __enter__(self):
            if hasattr(signal, 'SIGALRM'):  # Unix-like systems
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {self.timeout_seconds}s")
                
                self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, self.old_handler)
    
    return TimeoutContext


@pytest.fixture
def memory_monitor():
    """Memory usage monitoring for performance tests."""
    try:
        import psutil
        import os
        
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.initial_memory = None
                self.samples = []
            
            def start(self):
                self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.samples = [self.initial_memory]
                return self
            
            def sample(self):
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.samples.append(current_memory)
                return current_memory
            
            def get_peak(self):
                return max(self.samples) if self.samples else 0
            
            def get_growth(self):
                if len(self.samples) < 2:
                    return 0
                return max(self.samples) - self.samples[0]
        
        return MemoryMonitor
        
    except ImportError:
        # Fallback for systems without psutil
        class MockMemoryMonitor:
            def start(self):
                return self
            def sample(self):
                return 0
            def get_peak(self):
                return 0
            def get_growth(self):
                return 0
        
        return MockMemoryMonitor


@pytest.fixture
def profiler_context():
    """cProfile context manager for performance analysis."""
    import cProfile
    import pstats
    import io
    
    class ProfilerContext:
        def __init__(self):
            self.profiler = None
            self.stats = None
        
        def __enter__(self):
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.profiler.disable()
            s = io.StringIO()
            self.stats = pstats.Stats(self.profiler, stream=s)
            self.stats.sort_stats('cumulative')
        
        def get_stats(self, num_lines=10):
            if self.stats:
                s = io.StringIO()
                stats = pstats.Stats(self.profiler, stream=s)
                stats.sort_stats('cumulative')
                stats.print_stats(num_lines)
                return s.getvalue()
            return ""
        
        def get_function_time(self, function_name):
            """Get total time spent in functions containing function_name."""
            if not self.stats:
                return 0
            
            total_time = 0
            for func_info, (call_count, _, cumulative_time, _, _) in self.stats.stats.items():
                func_name = func_info[2]
                if function_name in func_name:
                    total_time += cumulative_time
            return total_time
    
    return ProfilerContext


@pytest.fixture
def performance_benchmark():
    """Performance benchmarking utilities."""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.benchmarks = {}
        
        def time_function(self, func, *args, **kwargs):
            """Time a single function call."""
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time
        
        def time_multiple(self, func, iterations=100, *args, **kwargs):
            """Time multiple iterations of a function."""
            times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            return {
                'times': times,
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
        
        def compare_functions(self, funcs_dict, iterations=100):
            """Compare performance of multiple functions."""
            results = {}
            for name, func_info in funcs_dict.items():
                func = func_info['func']
                args = func_info.get('args', ())
                kwargs = func_info.get('kwargs', {})
                results[name] = self.time_multiple(func, iterations, *args, **kwargs)
            return results
        
        def assert_performance_bounds(self, timing, max_time_ms, operation_name="Operation"):
            """Assert that timing is within performance bounds."""
            max_time_s = max_time_ms / 1000.0
            assert timing < max_time_s, f"{operation_name} took {timing*1000:.3f}ms, expected <{max_time_ms}ms"
        
        def assert_realtime_factor(self, render_time, audio_duration, max_factor=1.0):
            """Assert that audio rendering maintains realtime performance."""
            realtime_factor = render_time / audio_duration
            assert realtime_factor <= max_factor, f"Realtime factor {realtime_factor:.2f}x exceeds {max_factor}x"
    
    return PerformanceBenchmark


@pytest.fixture
def audio_validator():
    """Audio validation utilities."""
    class AudioValidator:
        @staticmethod
        def is_valid_range(audio_data, min_val=-1.0, max_val=1.0):
            """Check if audio data is within valid range."""
            return np.all((audio_data >= min_val) & (audio_data <= max_val))
        
        @staticmethod
        def is_not_silent(audio_data, threshold=1e-6):
            """Check if audio data contains non-silent content."""
            return np.any(np.abs(audio_data) > threshold)
        
        @staticmethod
        def has_expected_length(audio_data, sample_rate, duration, tolerance=0.01):
            """Check if audio data has expected length."""
            expected_samples = int(sample_rate * duration)
            actual_samples = len(audio_data)
            tolerance_samples = int(sample_rate * tolerance)
            return abs(actual_samples - expected_samples) <= tolerance_samples
        
        @staticmethod
        def compute_rms(audio_data):
            """Compute RMS level of audio data."""
            return np.sqrt(np.mean(audio_data ** 2))
        
        @staticmethod
        def compute_peak(audio_data):
            """Compute peak level of audio data."""
            return np.max(np.abs(audio_data))
        
        @staticmethod
        def has_no_clipping(audio_data, threshold=0.99):
            """Check if audio data has no clipping."""
            peak = AudioValidator.compute_peak(audio_data)
            return peak < threshold
    
    return AudioValidator


@pytest.fixture
def patch_validator():
    """Patch validation utilities."""
    class PatchValidator:
        @staticmethod
        def has_required_modules(patch_data, required_modules):
            """Check if patch has required modules."""
            modules = patch_data.get('modules', {})
            return all(module_id in modules for module_id in required_modules)
        
        @staticmethod
        def has_valid_connections(patch_data):
            """Check if patch has valid connections."""
            modules = patch_data.get('modules', {})
            connections = patch_data.get('connections', [])
            
            for conn in connections:
                source_module = conn['from'].split('.')[0]
                dest_module = conn['to'].split('.')[0]
                
                if source_module not in modules or dest_module not in modules:
                    return False
            
            return True
        
        @staticmethod
        def has_valid_sequence(patch_data):
            """Check if patch has valid sequence."""
            modules = patch_data.get('modules', {})
            sequence = patch_data.get('sequence', [])
            
            for event in sequence:
                if event.get('target') not in modules:
                    return False
            
            return True
    
    return PatchValidator


# Test data generators
@pytest.fixture
def parameter_variations():
    """Generate parameter variations for testing."""
    return {
        'frequencies': [110.0, 220.0, 440.0, 880.0, 1760.0],
        'amplitudes': [0.1, 0.5, 0.8, 1.0],
        'attack_times': [0.001, 0.01, 0.1, 0.5],
        'decay_times': [0.01, 0.1, 0.5, 1.0],
        'sustain_levels': [0.0, 0.3, 0.7, 1.0],
        'release_times': [0.01, 0.1, 0.5, 2.0],
        'gain_values': [0.0, 0.5, 1.0, 2.0]
    }


@pytest.fixture
def test_scenarios():
    """Define test scenarios for comprehensive testing."""
    return {
        'basic_oscillator': {
            'description': 'Basic oscillator test',
            'modules': ['oscillator'],
            'duration': 0.1,
            'expected_non_silent': True
        },
        'envelope_trigger': {
            'description': 'Envelope trigger test',
            'modules': ['envelope_adsr'],
            'duration': 1.0,
            'trigger_events': [{'time': 0.0, 'action': 'trigger'}],
            'expected_non_silent': True
        },
        'vca_modulation': {
            'description': 'VCA modulation test',
            'modules': ['oscillator', 'envelope_adsr', 'vca'],
            'duration': 1.0,
            'trigger_events': [{'time': 0.0, 'action': 'trigger'}],
            'expected_non_silent': True
        },
        'complex_synthesis': {
            'description': 'Complex synthesis pipeline',
            'modules': ['oscillator', 'oscillator', 'mixer', 'envelope_adsr', 'vca'],
            'duration': 2.0,
            'trigger_events': [
                {'time': 0.0, 'action': 'trigger'},
                {'time': 1.5, 'action': 'release'}
            ],
            'expected_non_silent': True
        }
    }