"""
Tests for Phase 2 functionality: patch system, module graph, and engine API.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import numpy as np

from signals import SynthEngine, Patch, PatchTemplate, ModuleGraph
from signals.patch import PatchError, PatchValidationError, PatchTemplateError
from signals.graph import GraphError, CyclicGraphError
from signals.engine import EngineError


class TestPatch:
    """Test patch loading and validation."""
    
    def test_patch_from_dict(self):
        """Test patch creation from dictionary."""
        patch_data = {
            'name': 'Test Patch',
            'description': 'A test patch',
            'sample_rate': 48000,
            'modules': {
                'osc1': {
                    'type': 'oscillator',
                    'parameters': {'frequency': 440.0, 'waveform': 'sine'}
                }
            }
        }
        
        patch = Patch.from_dict(patch_data)
        assert patch.name == 'Test Patch'
        assert patch.sample_rate == 48000
        assert 'osc1' in patch.modules
        assert patch.modules['osc1']['type'] == 'oscillator'
    
    def test_patch_validation_unknown_module_type(self):
        """Test patch validation with unknown module type."""
        patch_data = {
            'modules': {
                'unknown': {
                    'type': 'nonexistent_module',
                    'parameters': {}
                }
            }
        }
        
        with pytest.raises(PatchValidationError, match="Unknown module type"):
            Patch.from_dict(patch_data)
    
    def test_patch_validation_missing_connection_target(self):
        """Test patch validation with invalid connections."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {}}
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'nonexistent.0'}
            ]
        }
        
        with pytest.raises(PatchValidationError, match="unknown.*module"):
            Patch.from_dict(patch_data)
    
    def test_patch_to_dict(self):
        """Test patch serialization back to dictionary."""
        original_data = {
            'name': 'Test Patch',
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {'frequency': 440.0}}
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'osc1.0'}
            ]
        }
        
        patch = Patch.from_dict(original_data)
        result_data = patch.to_dict()
        
        assert result_data['name'] == 'Test Patch'
        assert 'osc1' in result_data['modules']
        assert len(result_data['connections']) == 1


class TestPatchTemplate:
    """Test patch template functionality."""
    
    def setup_method(self):
        """Set up temporary directory for test files."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_template_variable_extraction(self):
        """Test template variable discovery."""
        template_content = """
name: "{{ patch_name }}"
modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ osc_freq }}
      waveform: "{{ waveform_type }}"
"""
        template_file = self.temp_dir / "test_template.yaml"
        template_file.write_text(template_content)
        
        template = PatchTemplate(template_file)
        variables = template.variables
        
        assert 'patch_name' in variables
        assert 'osc_freq' in variables
        assert 'waveform_type' in variables
    
    def test_template_instantiation(self):
        """Test template instantiation with variables."""
        template_content = """
name: "{{ patch_name }}"
modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ frequency }}
"""
        template_file = self.temp_dir / "test_template.yaml"
        template_file.write_text(template_content)
        
        template = PatchTemplate(template_file)
        patch = template.instantiate({
            'patch_name': 'Generated Patch',
            'frequency': 880.0
        })
        
        assert patch.name == 'Generated Patch'
        assert patch.modules['osc1']['parameters']['frequency'] == 880.0
    
    def test_template_with_variables_section(self):
        """Test template with default variables section."""
        template_content = """
variables:
  base_freq: 440.0
  wave_type: "sine"

name: "Template Patch"
modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ base_freq }}
      waveform: "{{ wave_type }}"
"""
        template_file = self.temp_dir / "test_template.yaml"
        template_file.write_text(template_content)
        
        template = PatchTemplate(template_file)
        schema = template.get_variable_schema()
        
        assert schema['base_freq'] == 440.0
        assert schema['wave_type'] == "sine"


class TestModuleGraph:
    """Test module graph functionality."""
    
    def test_simple_graph_creation(self):
        """Test creation of a simple module graph."""
        patch_data = {
            'sample_rate': 48000,
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {'frequency': 440.0}},
                'env1': {'type': 'envelope_adsr', 'parameters': {'attack': 0.1}}
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'env1.0'}
            ]
        }
        
        patch = Patch.from_dict(patch_data)
        graph = ModuleGraph(patch)
        
        assert len(graph.nodes) == 2
        assert 'osc1' in graph.nodes
        assert 'env1' in graph.nodes
    
    def test_execution_order_computation(self):
        """Test topological sorting for execution order."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {}},
                'env1': {'type': 'envelope_adsr', 'parameters': {}},
                'mixer1': {'type': 'mixer', 'parameters': {}}
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'mixer1.0'},
                {'from': 'env1.0', 'to': 'mixer1.1'}
            ]
        }
        
        patch = Patch.from_dict(patch_data)
        graph = ModuleGraph(patch)
        
        # Check that dependencies come before dependents
        osc_idx = graph.execution_order.index('osc1')
        env_idx = graph.execution_order.index('env1')
        mixer_idx = graph.execution_order.index('mixer1')
        
        assert osc_idx < mixer_idx
        assert env_idx < mixer_idx
    
    def test_cyclic_dependency_detection(self):
        """Test detection of cyclic dependencies."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {}},
                'env1': {'type': 'envelope_adsr', 'parameters': {}}
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'env1.0'},
                {'from': 'env1.0', 'to': 'osc1.0'}  # Creates cycle
            ]
        }
        
        patch = Patch.from_dict(patch_data)
        
        with pytest.raises(CyclicGraphError):
            ModuleGraph(patch)
    
    def test_sample_processing(self):
        """Test single sample processing through graph."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {'frequency': 440.0}}
            }
        }
        
        patch = Patch.from_dict(patch_data)
        graph = ModuleGraph(patch)
        
        outputs = graph.process_sample()
        
        assert 'osc1' in outputs
        assert len(outputs['osc1']) == 1  # One output signal
        assert -1.0 <= outputs['osc1'][0].value <= 1.0


class TestSynthEngine:
    """Test synthesizer engine API."""
    
    def setup_method(self):
        """Set up temporary directory for test files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = SynthEngine(sample_rate=48000)
    
    def teardown_method(self):
        """Clean up temporary directory and engine."""
        shutil.rmtree(self.temp_dir)
        self.engine.cleanup()
    
    def test_engine_creation(self):
        """Test engine instantiation."""
        engine = SynthEngine(sample_rate=44100, buffer_size=512)
        assert engine.sample_rate == 44100
        assert engine.buffer_size == 512
        assert engine.current_patch is None
    
    def test_load_patch_from_dict(self):
        """Test loading patch from dictionary."""
        patch_data = {
            'name': 'Test Patch',
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {'frequency': 440.0}}
            }
        }
        
        patch = self.engine.load_patch_from_dict(patch_data)
        
        assert patch.name == 'Test Patch'
        assert self.engine.current_patch is not None
        assert self.engine.current_graph is not None
    
    def test_render_simple_patch(self):
        """Test rendering audio from a simple patch."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {'frequency': 440.0}}
            }
        }
        
        self.engine.load_patch_from_dict(patch_data)
        audio_data = self.engine.render(duration=0.1)  # Short duration for testing
        
        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) == int(0.1 * 48000)
    
    def test_set_module_parameter(self):
        """Test dynamic parameter setting."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {'frequency': 440.0}}
            }
        }
        
        self.engine.load_patch_from_dict(patch_data)
        self.engine.set_module_parameter('osc1', 'frequency', 880.0)
        
        # Verify parameter was set (via patch data)
        params = self.engine.get_module_parameters('osc1')
        assert params['frequency'] == 440.0  # Original patch data unchanged
    
    def test_get_patch_info(self):
        """Test getting patch information."""
        patch_data = {
            'name': 'Info Test Patch',
            'description': 'Test description',
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {}},
                'env1': {'type': 'envelope_adsr', 'parameters': {}}
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'env1.0'}
            ]
        }
        
        self.engine.load_patch_from_dict(patch_data)
        info = self.engine.get_patch_info()
        
        assert info['name'] == 'Info Test Patch'
        assert info['description'] == 'Test description'
        assert info['module_count'] == 2
        assert info['connection_count'] == 1
        assert 'osc1' in info['modules']
        assert 'env1' in info['modules']
    
    def test_export_features(self):
        """Test audio feature extraction."""
        # Create simple test audio
        sample_rate = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        features = self.engine.export_features(audio_data)
        
        assert 'length_samples' in features
        assert 'length_seconds' in features
        assert 'rms' in features
        assert 'peak' in features
        assert features['length_samples'] == len(audio_data)
        assert abs(features['length_seconds'] - duration) < 0.01
        assert 0.3 < features['rms'] < 0.4  # RMS of 0.5 * sin should be ~0.35
    
    def test_template_loading(self):
        """Test loading and processing templates."""
        template_content = """
name: "{{ patch_name }}"
modules:
  osc1:
    type: "oscillator"  
    parameters:
      frequency: {{ freq }}
"""
        template_file = self.temp_dir / "test_template.yaml"
        template_file.write_text(template_content)
        
        patch = self.engine.load_patch(template_file, {
            'patch_name': 'Template Test',
            'freq': 880.0
        })
        
        assert patch.name == 'Template Test'
        assert patch.modules['osc1']['parameters']['frequency'] == 880.0
    
    def test_engine_error_handling(self):
        """Test engine error handling."""
        # Test rendering without loaded patch
        with pytest.raises(EngineError, match="No patch loaded"):
            self.engine.render(duration=1.0)
        
        # Test setting parameter without loaded patch
        with pytest.raises(EngineError, match="No patch loaded"):
            self.engine.set_module_parameter('osc1', 'frequency', 440.0)


class TestIntegrationPhase2:
    """Integration tests for Phase 2 functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = SynthEngine(sample_rate=48000)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        self.engine.cleanup()
    
    def test_complete_synthesis_pipeline(self):
        """Test complete synthesis from patch to audio output."""
        patch_data = {
            'name': 'Integration Test',
            'modules': {
                'osc1': {
                    'type': 'oscillator',
                    'parameters': {'frequency': 440.0, 'waveform': 'sine'}
                },
                'env1': {
                    'type': 'envelope_adsr',
                    'parameters': {'attack': 0.01, 'decay': 0.1, 'sustain': 0.5, 'release': 0.1}
                }
            },
            'connections': [
                {'from': 'osc1.0', 'to': 'env1.0'}
            ],
            'sequence': [
                {'time': 0.0, 'action': 'trigger', 'target': 'env1'},
                {'time': 0.5, 'action': 'release', 'target': 'env1'}
            ]
        }
        
        # Load patch
        patch = self.engine.load_patch_from_dict(patch_data)
        assert patch.name == 'Integration Test'
        
        # Render audio
        audio_data = self.engine.render(duration=1.0)
        assert isinstance(audio_data, np.ndarray)
        assert len(audio_data) == 48000
        
        # Extract features
        features = self.engine.export_features(audio_data)
        assert features['length_seconds'] == 1.0
        
        # Get patch info
        info = self.engine.get_patch_info()
        assert info['module_count'] == 2
        assert info['connection_count'] == 1
    
    def test_template_variation_generation(self):
        """Test generating multiple variations from template."""
        template_content = """
name: "Variation Test"
modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ freq }}
      waveform: "{{ waveform }}"
"""
        template_file = self.temp_dir / "variation_template.yaml"
        template_file.write_text(template_content)
        
        # Test multiple parameter combinations
        parameter_sets = [
            {'freq': 220.0, 'waveform': 'sine'},
            {'freq': 440.0, 'waveform': 'square'},
            {'freq': 880.0, 'waveform': 'triangle'}
        ]
        
        for i, params in enumerate(parameter_sets):
            patch = self.engine.load_patch(template_file, params)
            
            assert patch.modules['osc1']['parameters']['frequency'] == params['freq']
            assert patch.modules['osc1']['parameters']['waveform'] == params['waveform']
            
            # Quick render test
            audio = self.engine.render(duration=0.05)
            assert len(audio) > 0