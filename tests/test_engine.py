"""
Comprehensive tests for the SynthEngine and patch system.

This test suite provides thorough testing of the synthesis engine,
patch loading, template processing, audio rendering, and integration
with the complete system.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import yaml
import time

from signals import SynthEngine, Patch, PatchTemplate, configure_logging, LogLevel
from signals.processing.patch import PatchError, PatchValidationError, PatchTemplateError
from signals.processing.engine import EngineError


@pytest.mark.unit
class TestSynthEngine:
    """Comprehensive tests for the SynthEngine class."""
    
    def test_engine_initialization(self, sample_rates):
        """Test engine initialization with different parameters."""
        for sample_rate in sample_rates:
            engine = SynthEngine(sample_rate=sample_rate, buffer_size=512)
            assert engine.sample_rate == sample_rate
            assert engine.buffer_size == 512
            assert engine.current_patch is None
            assert engine.current_graph is None
            engine.cleanup()
    
    def test_engine_load_patch_from_dict(self, synth_engine, basic_patch_data):
        """Test loading patch from dictionary."""
        with synth_engine() as engine:
            patch = engine.load_patch_from_dict(basic_patch_data)
            
            assert patch.name == basic_patch_data['name']
            assert engine.current_patch is not None
            assert engine.current_graph is not None
            assert len(engine.current_graph.nodes) == len(basic_patch_data['modules'])
    
    def test_engine_load_patch_from_file(self, synth_engine, basic_patch_data, 
                                        temp_dir, create_patch_file):
        """Test loading patch from file."""
        patch_file = create_patch_file(temp_dir, basic_patch_data)
        
        with synth_engine() as engine:
            patch = engine.load_patch(patch_file)
            assert patch.name == basic_patch_data['name']
            assert engine.current_patch is not None
    
    def test_engine_render_basic(self, synth_engine, basic_patch_data, audio_validator):
        """Test basic audio rendering."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            duration = 0.5
            audio_data = engine.render(duration=duration)
            
            # Validate audio output
            assert isinstance(audio_data, np.ndarray)
            assert audio_validator.has_expected_length(audio_data, engine.sample_rate, duration)
            assert audio_validator.is_valid_range(audio_data)
            assert audio_validator.has_no_clipping(audio_data)
    
    def test_engine_render_with_output_file(self, synth_engine, basic_patch_data, temp_dir):
        """Test rendering with file output."""
        output_file = temp_dir / "test_render.wav"
        
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            audio_data = engine.render(duration=0.2, output_file=str(output_file))
            
            assert isinstance(audio_data, np.ndarray)
            assert output_file.exists()
            assert output_file.stat().st_size > 1000  # Should have reasonable size
    
    def test_engine_auto_duration_calculation(self, synth_engine, basic_patch_data):
        """Test automatic duration calculation."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            # Render without specifying duration (should auto-calculate)
            audio_data = engine.render()
            
            assert len(audio_data) > 0
            # Duration should be based on sequence + envelope release
            expected_min_samples = int(engine.sample_rate * 1.0)  # At least 1 second
            assert len(audio_data) >= expected_min_samples
    
    def test_engine_set_module_parameter(self, synth_engine, basic_patch_data):
        """Test dynamic parameter setting."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            # Set oscillator frequency
            engine.set_module_parameter('osc1', 'frequency', 880.0)
            
            # Set envelope parameters
            engine.set_module_parameter('env1', 'attack', 0.1)
            engine.set_module_parameter('env1', 'release', '30%')
            
            # Render to verify parameters took effect
            audio_data = engine.render(duration=0.1)
            assert len(audio_data) > 0
    
    def test_engine_get_module_parameters(self, synth_engine, basic_patch_data):
        """Test getting module parameters."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            # Get oscillator parameters
            osc_params = engine.get_module_parameters('osc1')
            assert 'frequency' in osc_params
            assert 'waveform' in osc_params
            assert 'amplitude' in osc_params
            
            # Get envelope parameters
            env_params = engine.get_module_parameters('env1')
            assert 'attack' in env_params
            assert 'decay' in env_params
            assert 'sustain' in env_params
            assert 'release' in env_params
    
    def test_engine_get_patch_info(self, synth_engine, basic_patch_data):
        """Test getting patch information."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            info = engine.get_patch_info()
            
            assert info['name'] == basic_patch_data['name']
            assert info['description'] == basic_patch_data['description']
            assert info['sample_rate'] == basic_patch_data['sample_rate']
            assert info['module_count'] == len(basic_patch_data['modules'])
            assert info['connection_count'] == len(basic_patch_data['connections'])
            assert 'modules' in info
            assert 'execution_order' in info
    
    def test_engine_export_features(self, synth_engine, sample_audio_data):
        """Test audio feature extraction."""
        audio_data = sample_audio_data(duration=1.0, frequency=440.0)
        
        with synth_engine() as engine:
            features = engine.export_features(audio_data)
            
            assert 'length_samples' in features
            assert 'length_seconds' in features
            assert 'rms' in features
            assert 'peak' in features
            assert 'zero_crossings' in features
            
            # Validate feature values
            assert features['length_samples'] == len(audio_data)
            assert abs(features['length_seconds'] - 1.0) < 0.01
            assert 0.1 < features['rms'] < 0.5  # Should have reasonable RMS
            assert 0.3 < features['peak'] < 0.6  # Should have reasonable peak
    
    def test_engine_error_handling(self, synth_engine):
        """Test engine error handling."""
        with synth_engine() as engine:
            # Test rendering without loaded patch
            with pytest.raises(EngineError, match="No patch loaded"):
                engine.render(duration=1.0)
            
            # Test setting parameter without loaded patch
            with pytest.raises(EngineError, match="No patch loaded"):
                engine.set_module_parameter('osc1', 'frequency', 440.0)
            
            # Test getting parameters for non-existent module
            engine.load_patch_from_dict({'modules': {'osc1': {'type': 'oscillator', 'parameters': {}}}})
            
            with pytest.raises(EngineError, match="not found"):
                engine.get_module_parameters('non_existent')
    
    def test_engine_template_loading(self, synth_engine, template_patch_content, 
                                   temp_dir, create_template_file):
        """Test loading and processing templates."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        variables = {
            'patch_name': 'Template Test',
            'osc_freq': 880.0,
            'waveform': 'square',
            'env_attack': 0.1
        }
        
        with synth_engine() as engine:
            patch = engine.load_patch(template_file, variables)
            
            assert patch.name == 'Template Test'
            assert patch.modules['osc1']['parameters']['frequency'] == 880.0
            assert patch.modules['osc1']['parameters']['waveform'] == 'square'
            assert patch.modules['env1']['parameters']['attack'] == 0.1


@pytest.mark.unit
class TestPatch:
    """Comprehensive tests for the Patch class."""
    
    def test_patch_from_dict_basic(self, basic_patch_data):
        """Test basic patch creation from dictionary."""
        patch = Patch.from_dict(basic_patch_data)
        
        assert patch.name == basic_patch_data['name']
        assert patch.description == basic_patch_data['description']
        assert patch.sample_rate == basic_patch_data['sample_rate']
        assert len(patch.modules) == len(basic_patch_data['modules'])
        assert len(patch.connections) == len(basic_patch_data['connections'])
        assert len(patch.sequence) == len(basic_patch_data['sequence'])
    
    def test_patch_from_file(self, basic_patch_data, temp_dir, create_patch_file):
        """Test patch loading from file."""
        patch_file = create_patch_file(temp_dir, basic_patch_data)
        
        patch = Patch.from_file(patch_file)
        assert patch.name == basic_patch_data['name']
        assert patch.source_file == patch_file
    
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
    
    def test_patch_validation_missing_sequence_target(self):
        """Test patch validation with invalid sequence targets."""
        patch_data = {
            'modules': {
                'osc1': {'type': 'oscillator', 'parameters': {}}
            },
            'sequence': [
                {'time': 0.0, 'action': 'trigger', 'target': 'nonexistent'}
            ]
        }
        
        with pytest.raises(PatchValidationError, match="unknown module"):
            Patch.from_dict(patch_data)
    
    def test_patch_to_dict_roundtrip(self, basic_patch_data):
        """Test patch serialization roundtrip."""
        original_patch = Patch.from_dict(basic_patch_data)
        serialized_data = original_patch.to_dict()
        reconstructed_patch = Patch.from_dict(serialized_data)
        
        assert original_patch.name == reconstructed_patch.name
        assert original_patch.sample_rate == reconstructed_patch.sample_rate
        assert len(original_patch.modules) == len(reconstructed_patch.modules)
        assert len(original_patch.connections) == len(reconstructed_patch.connections)
    
    def test_patch_get_duration(self, basic_patch_data):
        """Test patch duration calculation."""
        patch = Patch.from_dict(basic_patch_data)
        duration = patch.get_duration()
        
        # Duration should be the maximum time in sequence
        expected_duration = max(event['time'] for event in basic_patch_data['sequence'])
        assert duration == expected_duration
    
    def test_patch_empty_sequence(self):
        """Test patch with empty sequence."""
        patch_data = {
            'modules': {'osc1': {'type': 'oscillator', 'parameters': {}}},
            'sequence': []
        }
        
        patch = Patch.from_dict(patch_data)
        assert patch.get_duration() == 0.0
    
    def test_patch_module_count(self, basic_patch_data):
        """Test patch module counting."""
        patch = Patch.from_dict(basic_patch_data)
        assert patch.get_module_count() == len(basic_patch_data['modules'])
    
    def test_patch_connection_count(self, basic_patch_data):
        """Test patch connection counting."""
        patch = Patch.from_dict(basic_patch_data)
        assert patch.get_connection_count() == len(basic_patch_data['connections'])


@pytest.mark.unit
class TestPatchTemplate:
    """Comprehensive tests for the PatchTemplate class."""
    
    def test_template_variable_extraction(self, template_patch_content, 
                                        temp_dir, create_template_file):
        """Test template variable discovery."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        template = PatchTemplate(template_file)
        variables = template.variables
        
        expected_vars = ['patch_name', 'description', 'sample_rate', 'osc_freq', 
                        'waveform', 'amplitude', 'env_attack', 'env_decay', 
                        'env_sustain', 'env_release', 'vca_gain', 'trigger_time', 
                        'release_time']
        
        for var in expected_vars:
            assert var in variables
    
    def test_template_get_variable_schema(self, template_patch_content, 
                                        temp_dir, create_template_file):
        """Test template variable schema extraction."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        template = PatchTemplate(template_file)
        schema = template.get_variable_schema()
        
        # Should extract default values from variables section
        assert 'osc_freq' in schema
        assert schema['osc_freq'] == 440.0
        assert 'env_attack' in schema
        assert schema['env_attack'] == 0.02
    
    def test_template_instantiation_basic(self, template_patch_content, 
                                        temp_dir, create_template_file):
        """Test basic template instantiation."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        template = PatchTemplate(template_file)
        patch = template.instantiate({
            'patch_name': 'Instantiated Patch',
            'osc_freq': 880.0,
            'waveform': 'square'
        })
        
        assert patch.name == 'Instantiated Patch'
        assert patch.modules['osc1']['parameters']['frequency'] == 880.0
        assert patch.modules['osc1']['parameters']['waveform'] == 'square'
    
    def test_template_instantiation_with_defaults(self, template_patch_content, 
                                                temp_dir, create_template_file):
        """Test template instantiation using default values."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        template = PatchTemplate(template_file)
        patch = template.instantiate({})  # No variables provided
        
        # Should use default values
        assert patch.modules['osc1']['parameters']['frequency'] == 440.0  # Default
        assert patch.modules['env1']['parameters']['attack'] == 0.02  # Default
    
    def test_template_instantiation_partial_variables(self, template_patch_content, 
                                                    temp_dir, create_template_file):
        """Test template instantiation with partial variables."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        template = PatchTemplate(template_file)
        patch = template.instantiate({
            'osc_freq': 1760.0,
            'env_attack': 0.2
            # Other variables should use defaults
        })
        
        assert patch.modules['osc1']['parameters']['frequency'] == 1760.0
        assert patch.modules['env1']['parameters']['attack'] == 0.2
        assert patch.modules['env1']['parameters']['release'] == 0.2  # Default
    
    def test_template_file_not_found(self, temp_dir):
        """Test template loading with non-existent file."""
        non_existent_file = temp_dir / "does_not_exist.yaml"
        
        with pytest.raises(PatchTemplateError, match="not found"):
            PatchTemplate(non_existent_file)
    
    def test_template_invalid_syntax(self, temp_dir):
        """Test template with invalid Jinja2 syntax."""
        invalid_template = """
        name: "{{ unclosed_variable"
        modules:
          osc1:
            type: "oscillator"
        """
        
        template_file = temp_dir / "invalid.yaml"
        template_file.write_text(invalid_template)
        
        template = PatchTemplate(template_file)
        
        with pytest.raises(PatchTemplateError, match="Template rendering error"):
            template.instantiate({})


@pytest.mark.integration
class TestEngineIntegration:
    """Integration tests for complete engine functionality."""
    
    def test_complete_synthesis_pipeline(self, synth_engine, complex_patch_data, 
                                       audio_validator):
        """Test complete synthesis from patch to audio output."""
        with synth_engine() as engine:
            # Load complex patch
            patch = engine.load_patch_from_dict(complex_patch_data)
            assert patch.name == complex_patch_data['name']
            
            # Render audio
            duration = 1.0
            audio_data = engine.render(duration=duration)
            
            # Validate audio output
            assert audio_validator.has_expected_length(audio_data, engine.sample_rate, duration)
            assert audio_validator.is_valid_range(audio_data)
            assert audio_validator.is_not_silent(audio_data)
            assert audio_validator.has_no_clipping(audio_data)
            
            # Extract features
            features = engine.export_features(audio_data)
            assert features['length_seconds'] == duration
            
            # Get patch info
            info = engine.get_patch_info()
            assert info['module_count'] == len(complex_patch_data['modules'])
            assert info['connection_count'] == len(complex_patch_data['connections'])
    
    def test_template_batch_processing(self, synth_engine, template_patch_content, 
                                     temp_dir, create_template_file, audio_validator):
        """Test batch processing of template variations."""
        template_file = create_template_file(temp_dir, template_patch_content)
        
        parameter_sets = [
            {'osc_freq': 220.0, 'waveform': 'sine', 'env_attack': 0.05},
            {'osc_freq': 440.0, 'waveform': 'square', 'env_attack': 0.1},
            {'osc_freq': 880.0, 'waveform': 'saw', 'env_attack': 0.2}
        ]
        
        with synth_engine() as engine:
            results = []
            for i, params in enumerate(parameter_sets):
                # Load template with parameters
                patch = engine.load_patch(template_file, params)
                assert patch.modules['osc1']['parameters']['frequency'] == params['osc_freq']
                
                # Render audio
                audio_data = engine.render(duration=0.5)
                
                # Validate
                assert audio_validator.is_valid_range(audio_data)
                assert audio_validator.is_not_silent(audio_data)
                
                # Extract features
                features = engine.export_features(audio_data)
                results.append({
                    'parameters': params,
                    'features': features,
                    'audio_length': len(audio_data)
                })
            
            # Verify all variations were processed
            assert len(results) == len(parameter_sets)
            
            # Verify different frequencies produced different results
            rms_values = [r['features']['rms'] for r in results]
            assert len(set(np.round(rms_values, 3))) > 1  # Should have variation
    
    def test_dynamic_parameter_changes(self, synth_engine, basic_patch_data, audio_validator):
        """Test dynamic parameter changes during operation."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            # Test multiple parameter changes
            test_frequencies = [220.0, 440.0, 880.0, 1760.0]
            test_waveforms = ['sine', 'square', 'saw', 'triangle']
            
            for freq, waveform in zip(test_frequencies, test_waveforms):
                # Change parameters
                engine.set_module_parameter('osc1', 'frequency', freq)
                engine.set_module_parameter('osc1', 'waveform', waveform)
                engine.set_module_parameter('env1', 'attack', freq / 10000.0)  # Proportional attack
                
                # Render short audio segment
                audio_data = engine.render(duration=0.1)
                
                # Validate each segment
                assert audio_validator.is_valid_range(audio_data)
                assert len(audio_data) > 0
    
    def test_engine_performance_characteristics(self, synth_engine, complex_patch_data, 
                                              performance_timer):
        """Test engine performance characteristics."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(complex_patch_data)
            
            # Test rendering performance
            durations = [0.1, 0.5, 1.0, 2.0]
            timing_results = []
            
            for duration in durations:
                with performance_timer() as timer:
                    audio_data = engine.render(duration=duration)
                
                timing_results.append({
                    'duration': duration,
                    'samples': len(audio_data),
                    'render_time': timer.elapsed,
                    'realtime_ratio': timer.elapsed / duration
                })
            
            # Verify reasonable performance
            for result in timing_results:
                # Should render faster than real-time for most cases
                assert result['realtime_ratio'] < 10.0  # At most 10x real-time
                assert result['samples'] > 0
    
    def test_error_recovery(self, synth_engine, basic_patch_data):
        """Test engine error recovery capabilities."""
        with synth_engine() as engine:
            # Load valid patch
            engine.load_patch_from_dict(basic_patch_data)
            audio_data = engine.render(duration=0.1)
            assert len(audio_data) > 0
            
            # Try invalid operations
            try:
                engine.set_module_parameter('non_existent', 'param', 123)
            except EngineError:
                pass
            
            try:
                engine.set_module_parameter('osc1', 'invalid_param', 123)
            except:
                pass
            
            # Engine should still work after errors
            audio_data = engine.render(duration=0.1)
            assert len(audio_data) > 0
    
    def test_memory_cleanup(self, synth_engine, basic_patch_data):
        """Test proper memory cleanup."""
        with synth_engine() as engine:
            # Load and process multiple patches
            for i in range(5):
                modified_patch = basic_patch_data.copy()
                modified_patch['name'] = f'Test Patch {i}'
                modified_patch['modules']['osc1']['parameters']['frequency'] = 440.0 * (i + 1)
                
                engine.load_patch_from_dict(modified_patch)
                audio_data = engine.render(duration=0.1)
                assert len(audio_data) > 0
            
            # Final cleanup should not raise errors
            # (cleanup is handled by context manager)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for engine and patch system."""
    
    def test_render_performance_scaling(self, synth_engine, basic_patch_data, 
                                      performance_timer):
        """Test render performance scaling with duration."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            durations = [0.1, 0.2, 0.4, 0.8]
            times = []
            
            for duration in durations:
                with performance_timer() as timer:
                    audio_data = engine.render(duration=duration)
                
                times.append(timer.elapsed)
                assert len(audio_data) == int(duration * engine.sample_rate)
            
            # Rendering time should scale roughly linearly with duration
            # Allow some overhead for setup
            for i in range(1, len(times)):
                ratio = times[i] / times[0]
                duration_ratio = durations[i] / durations[0]
                assert ratio < duration_ratio * 2.0  # Within 2x of linear scaling
    
    def test_patch_loading_performance(self, synth_engine, complex_patch_data, 
                                     performance_timer):
        """Test patch loading performance."""
        with synth_engine() as engine:
            loading_times = []
            
            for _ in range(10):
                with performance_timer() as timer:
                    engine.load_patch_from_dict(complex_patch_data)
                
                loading_times.append(timer.elapsed)
            
            # Patch loading should be fast and consistent
            avg_time = sum(loading_times) / len(loading_times)
            max_time = max(loading_times)
            
            assert avg_time < 0.1  # Should load in less than 100ms
            assert max_time < 0.2  # No load should take more than 200ms
    
    def test_parameter_change_performance(self, synth_engine, basic_patch_data, 
                                        performance_timer):
        """Test parameter change performance."""
        with synth_engine() as engine:
            engine.load_patch_from_dict(basic_patch_data)
            
            # Test many rapid parameter changes
            with performance_timer() as timer:
                for i in range(1000):
                    freq = 220.0 + (i % 100) * 10.0
                    engine.set_module_parameter('osc1', 'frequency', freq)
            
            # Should handle rapid parameter changes efficiently
            assert timer.elapsed < 1.0  # 1000 changes in less than 1 second