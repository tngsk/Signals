"""
Debugging and profiling tests for the Signals synthesizer framework.

These tests help identify performance bottlenecks, memory issues, and provide
detailed analysis capabilities for development and optimization.
"""

import pytest
import time
import tempfile
import cProfile
import pstats
import io
import gc
from pathlib import Path
from unittest.mock import patch
import numpy as np

from signals import SynthEngine, Patch, Oscillator, EnvelopeADSR, Mixer, Signal, SignalType, ModuleGraph


@pytest.fixture
def profiling_patch_content():
    """Patch configuration optimized for profiling tests."""
    return """name: "Profiling Test Patch"
description: "Patch designed for detailed profiling analysis"

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: 440.0
      waveform: "sine"
      
  osc2:
    type: "oscillator"  
    parameters:
      frequency: 660.0
      waveform: "square"
      
  osc3:
    type: "oscillator"
    parameters:
      frequency: 220.0
      waveform: "saw"
      
  mixer1:
    type: "mixer"
    parameters:
      gain1: 0.5
      gain2: 0.3
      
  mixer2:
    type: "mixer"
    parameters:
      gain1: 0.7
      gain2: 0.4
      
  env1:
    type: "envelope_adsr"
    parameters:
      attack: 0.05
      decay: 0.2
      sustain: 0.6
      release: 0.3
      
  env2:
    type: "envelope_adsr"
    parameters:
      attack: 0.1
      decay: 0.3
      sustain: 0.4
      release: 0.2

connections:
  - from: "osc1.0"
    to: "mixer1.0"
  - from: "osc2.0"
    to: "mixer1.1"
  - from: "osc3.0"
    to: "mixer2.0"
  - from: "mixer1.0"
    to: "mixer2.1"
  - from: "mixer2.0"
    to: "env1.0"
  - from: "env1.0"
    to: "env2.0"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 0.1
    action: "trigger"
    target: "env2"
  - time: 1.0
    action: "release"
    target: "env1"
  - time: 1.5
    action: "release"
    target: "env2"
"""


@pytest.fixture
def temp_profiling_patch(profiling_patch_content):
    """Create temporary patch file for profiling."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(profiling_patch_content)
        temp_path = f.name
    
    yield temp_path
    
    try:
        Path(temp_path).unlink()
    except FileNotFoundError:
        pass


@pytest.mark.performance
class TestModuleProfiling:
    """Detailed profiling tests for individual modules."""
    
    def test_oscillator_detailed_profiling(self):
        """Profile oscillator processing with different configurations."""
        sample_rate = 48000
        test_configs = [
            {"frequency": 440.0, "waveform": "sine"},
            {"frequency": 880.0, "waveform": "square"},
            {"frequency": 220.0, "waveform": "triangle"},
            {"frequency": 1760.0, "waveform": "saw"},
        ]
        
        profiling_results = {}
        
        for config in test_configs:
            osc = Oscillator(sample_rate)
            for param, value in config.items():
                osc.set_parameter(param, value)
            
            # Warm up
            for _ in range(10):
                osc.process()
            
            # Profile processing
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.perf_counter()
            for _ in range(1000):
                output = osc.process()
                assert len(output) == 1
                assert output[0].type == SignalType.AUDIO
            end_time = time.perf_counter()
            
            profiler.disable()
            
            # Analyze results
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            
            config_key = f"{config['waveform']}_{config['frequency']}"
            profiling_results[config_key] = {
                'total_time': end_time - start_time,
                'avg_time': (end_time - start_time) / 1000,
                'stats': stats
            }
        
        # Verify all configurations perform reasonably
        for config_key, results in profiling_results.items():
            avg_time = results['avg_time']
            assert avg_time < 0.00005, f"{config_key}: {avg_time*1000:.3f}ms too slow"
    
    def test_envelope_phase_profiling(self):
        """Profile envelope processing through different phases."""
        sample_rate = 48000
        env = EnvelopeADSR(sample_rate)
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.2)
        env.set_parameter("sustain", 0.6)
        env.set_parameter("release", 0.3)
        
        phases = {
            'idle': (lambda: None, 100),
            'attack': (lambda: env.trigger_on(), 200),
            'sustain': (lambda: None, 100),
            'release': (lambda: env.trigger_off(), 200)
        }
        
        phase_timings = {}
        
        for phase_name, (setup_func, sample_count) in phases.items():
            if setup_func:
                setup_func()
            
            # Let envelope settle into phase
            for _ in range(10):
                env.process()
            
            # Profile this phase
            start_time = time.perf_counter()
            for _ in range(sample_count):
                output = env.process()
                assert len(output) == 1
                assert output[0].type == SignalType.CONTROL
                assert 0.0 <= output[0].value <= 1.0
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / sample_count
            phase_timings[phase_name] = avg_time
        
        # All phases should be fast
        for phase, timing in phase_timings.items():
            assert timing < 0.00002, f"{phase} phase too slow: {timing*1000:.3f}ms"
        
        # Performance should be consistent across phases
        min_time = min(phase_timings.values())
        max_time = max(phase_timings.values())
        variance_ratio = max_time / min_time
        assert variance_ratio < 5.0, f"Excessive phase performance variance: {variance_ratio:.1f}x"
    
    def test_mixer_scaling_profiling(self):
        """Profile mixer performance with different input counts."""
        input_counts = [2, 4, 8, 16]
        scaling_results = {}
        
        for input_count in input_counts:
            mixer = Mixer(num_inputs=input_count)
            
            # Set gains for all inputs
            for i in range(input_count):
                mixer.set_parameter(f"gain{i+1}", 1.0 / input_count)
            
            # Create test inputs
            inputs = [Signal(SignalType.AUDIO, 0.1) for _ in range(input_count)]
            
            # Profile processing
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.perf_counter()
            for _ in range(1000):
                output = mixer.process(inputs)
                assert len(output) == 1
                assert output[0].type == SignalType.AUDIO
            end_time = time.perf_counter()
            
            profiler.disable()
            
            avg_time = (end_time - start_time) / 1000
            scaling_results[input_count] = avg_time
        
        # Verify linear or sub-linear scaling
        base_time = scaling_results[2]
        for input_count, timing in scaling_results.items():
            if input_count > 2:
                scaling_factor = timing / base_time
                theoretical_max = input_count / 2  # Linear scaling
                assert scaling_factor <= theoretical_max * 2, f"{input_count} inputs: {scaling_factor:.1f}x slower than expected"


@pytest.mark.performance  
class TestGraphProfiling:
    """Detailed profiling of ModuleGraph processing."""
    
    def test_graph_creation_profiling(self, temp_profiling_patch):
        """Profile graph creation and initialization."""
        patch = Patch.from_file(temp_profiling_patch)
        
        # Profile graph creation
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.perf_counter()
        graph = ModuleGraph(patch)
        creation_time = time.perf_counter() - start_time
        
        profiler.disable()
        
        # Analyze profiling results
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        
        # Graph creation should be fast
        assert creation_time < 0.1, f"Graph creation took {creation_time:.3f}s"
        
        # Verify graph structure
        assert len(graph.nodes) == len(patch.modules)
        assert len(graph.execution_order) == len(patch.modules)
    
    def test_single_sample_profiling(self, temp_profiling_patch):
        """Profile single sample processing with detailed breakdown."""
        patch = Patch.from_file(temp_profiling_patch)
        graph = ModuleGraph(patch)
        
        # Warm up
        for _ in range(10):
            graph.process_sample()
        
        # Profile sample processing
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.perf_counter()
        for _ in range(100):
            outputs = graph.process_sample()
            assert len(outputs) > 0
        end_time = time.perf_counter()
        
        profiler.disable()
        
        # Analyze timing
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Should process samples efficiently
        assert avg_time < 0.001, f"Sample processing too slow: {avg_time*1000:.3f}ms"
        
        # Generate profiling report
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        # Look for potential hotspots in the output
        report = s.getvalue()
        lines = report.split('\n')
        
        # Check for excessive calls to common functions
        for line in lines[:10]:  # Top 10 functions
            if 'process' in line and 'signals' in line:
                # Extract call count and time info if available
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        call_count = int(parts[0])
                        # Shouldn't have excessive redundant calls
                        assert call_count < 10000, f"Excessive calls detected: {line}"
                    except ValueError:
                        pass  # Skip lines that don't start with call count
    
    def test_step_by_step_profiling(self, temp_profiling_patch):
        """Profile each step of graph processing individually."""
        patch = Patch.from_file(temp_profiling_patch)
        graph = ModuleGraph(patch)
        
        execution_timings = {}
        
        # Reset all nodes
        for node in graph.nodes.values():
            node.reset_cycle()
        
        # Process each module and time it
        outputs = {}
        for module_id in graph.execution_order:
            node = graph.nodes[module_id]
            
            # Gather inputs
            inputs = []
            for input_idx in range(node.module.input_count):
                if input_idx in node.input_connections:
                    source_id, output_idx = node.input_connections[input_idx]
                    if source_id in outputs and output_idx < len(outputs[source_id]):
                        inputs.append(outputs[source_id][output_idx])
                    else:
                        inputs.append(Signal(SignalType.AUDIO, 0.0))
                else:
                    inputs.append(Signal(SignalType.AUDIO, 0.0))
            
            # Time module processing
            start_time = time.perf_counter()
            if inputs or node.module.input_count == 0:
                node_outputs = node.module.process(inputs if inputs else None)
            else:
                node_outputs = node.module.process()
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            execution_timings[module_id] = processing_time
            
            outputs[module_id] = node_outputs
            node.cached_outputs = node_outputs
            node.processed_this_cycle = True
        
        # Analyze timings
        total_time = sum(execution_timings.values())
        
        for module_id, timing in execution_timings.items():
            # Individual modules should be fast
            assert timing < 0.0001, f"{module_id} too slow: {timing*1000:.3f}ms"
            
            # No module should dominate processing time excessively
            percentage = (timing / total_time) * 100 if total_time > 0 else 0
            assert percentage < 80, f"{module_id} takes {percentage:.1f}% of processing time"


@pytest.mark.performance
class TestMemoryProfiling:
    """Memory usage profiling and leak detection."""
    
    def test_memory_usage_baseline(self):
        """Establish baseline memory usage for modules."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
        except ImportError:
            pytest.skip("psutil not available for memory profiling")
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create modules
        osc = Oscillator(48000)
        after_osc = process.memory_info().rss / 1024 / 1024
        
        env = EnvelopeADSR(48000)
        after_env = process.memory_info().rss / 1024 / 1024
        
        mixer = Mixer(num_inputs=4)
        after_mixer = process.memory_info().rss / 1024 / 1024
        
        # Memory increases should be reasonable
        osc_memory = after_osc - initial_memory
        env_memory = after_env - after_osc
        mixer_memory = after_mixer - after_env
        
        assert osc_memory < 10, f"Oscillator uses {osc_memory:.1f}MB"
        assert env_memory < 10, f"Envelope uses {env_memory:.1f}MB" 
        assert mixer_memory < 10, f"Mixer uses {mixer_memory:.1f}MB"
    
    def test_processing_memory_stability(self, temp_profiling_patch):
        """Test for memory leaks during processing."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
        except ImportError:
            pytest.skip("psutil not available for memory profiling")
        
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_profiling_patch)
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        memory_samples = []
        
        # Process many samples and track memory
        for i in range(200):
            audio = engine.render(duration=0.01)  # 10ms renders
            assert len(audio) > 0
            
            if i % 20 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        # Memory should not grow excessively
        max_memory = max(memory_samples)
        memory_growth = max_memory - baseline_memory
        
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB"
        
        # Memory should be relatively stable (not continuously growing)
        if len(memory_samples) >= 5:
            early_avg = sum(memory_samples[:3]) / 3
            late_avg = sum(memory_samples[-3:]) / 3
            growth_rate = (late_avg - early_avg) / early_avg * 100
            
            assert growth_rate < 20, f"Memory growing at {growth_rate:.1f}% rate"
    
    def test_object_creation_profiling(self, temp_profiling_patch):
        """Profile object creation and garbage collection."""
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_profiling_patch)
        
        # Track object counts
        initial_objects = len(gc.get_objects())
        
        # Process several renders
        for _ in range(50):
            audio = engine.render(duration=0.01)
            assert len(audio) > 0
        
        before_gc_objects = len(gc.get_objects())
        
        # Force garbage collection
        collected = gc.collect()
        
        after_gc_objects = len(gc.get_objects())
        
        # Analyze object growth
        object_growth = before_gc_objects - initial_objects
        objects_collected = before_gc_objects - after_gc_objects
        
        # Should not create excessive temporary objects
        assert object_growth < 10000, f"Created {object_growth} objects"
        
        # Garbage collection should be effective if objects were created
        if object_growth > 100:
            collection_rate = objects_collected / object_growth * 100
            assert collection_rate > 50, f"Only {collection_rate:.1f}% objects collected"


@pytest.mark.performance
class TestComparativeProfiling:
    """Comparative performance analysis."""
    
    def test_graph_vs_manual_processing(self, temp_profiling_patch):
        """Compare ModuleGraph performance vs manual processing."""
        # Setup ModuleGraph approach
        engine = SynthEngine(sample_rate=48000)
        patch = engine.load_patch(temp_profiling_patch)
        
        # Setup manual processing
        osc1 = Oscillator(48000)
        osc1.set_parameter("frequency", 440.0)
        osc1.set_parameter("waveform", "sine")
        
        osc2 = Oscillator(48000)
        osc2.set_parameter("frequency", 660.0)
        osc2.set_parameter("waveform", "square")
        
        mixer = Mixer(num_inputs=2)
        mixer.set_parameter("gain1", 0.7)
        mixer.set_parameter("gain2", 0.3)
        
        env = EnvelopeADSR(48000)
        env.set_parameter("attack", 0.1)
        env.trigger_on()
        
        def manual_process():
            osc1_out = osc1.process()
            osc2_out = osc2.process()
            mixer_out = mixer.process([osc1_out[0], osc2_out[0]])
            env_out = env.process([mixer_out[0]])
            return env_out
        
        # Benchmark ModuleGraph
        graph_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            graph_output = engine.current_graph.process_sample()
            end_time = time.perf_counter()
            graph_times.append(end_time - start_time)
        
        # Benchmark manual processing
        manual_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            manual_output = manual_process()
            end_time = time.perf_counter()
            manual_times.append(end_time - start_time)
        
        # Analyze performance
        avg_graph_time = sum(graph_times) / len(graph_times)
        avg_manual_time = sum(manual_times) / len(manual_times)
        
        overhead_factor = avg_graph_time / avg_manual_time
        
        # Graph overhead should be reasonable (less than 10x slower)
        assert overhead_factor < 10, f"Graph {overhead_factor:.1f}x slower than manual"
        
        # Both should be fast enough for realtime
        assert avg_graph_time < 0.001, f"Graph processing too slow: {avg_graph_time*1000:.3f}ms"
        assert avg_manual_time < 0.0001, f"Manual processing too slow: {avg_manual_time*1000:.3f}ms"
    
    def test_parameter_change_profiling(self):
        """Profile performance impact of parameter changes."""
        osc = Oscillator(48000)
        
        # Baseline processing time
        baseline_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            osc.process()
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Processing with frequent parameter changes
        change_times = []
        frequencies = [220, 440, 880, 1760]
        freq_idx = 0
        
        for i in range(100):
            if i % 10 == 0:  # Change parameter every 10 samples
                osc.set_parameter("frequency", frequencies[freq_idx])
                freq_idx = (freq_idx + 1) % len(frequencies)
            
            start_time = time.perf_counter()
            osc.process()
            end_time = time.perf_counter()
            change_times.append(end_time - start_time)
        
        change_avg = sum(change_times) / len(change_times)
        
        # Parameter changes shouldn't cause excessive overhead
        overhead_factor = change_avg / baseline_avg
        assert overhead_factor < 5, f"Parameter changes cause {overhead_factor:.1f}x overhead"


@pytest.mark.slow
@pytest.mark.performance
class TestLongRunningProfiling:
    """Long-running profiling tests for stability analysis."""
    
    def test_extended_processing_stability(self, temp_profiling_patch):
        """Test processing stability over extended periods."""
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_profiling_patch)
        
        sample_times = []
        
        # Process for equivalent of 10 seconds of audio
        total_samples = 10 * 48000
        batch_size = 4800  # Process in 0.1s batches
        
        for batch in range(total_samples // batch_size):
            start_time = time.perf_counter()
            audio = engine.render(duration=batch_size / 48000)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            per_sample_time = batch_time / batch_size
            sample_times.append(per_sample_time)
            
            # Verify audio quality
            assert len(audio) == batch_size
            assert np.max(np.abs(audio)) > 0.001, f"Batch {batch} audio too quiet"
            assert not np.any(np.isnan(audio)), f"Batch {batch} contains NaN"
            assert not np.any(np.isinf(audio)), f"Batch {batch} contains inf"
        
        # Analyze performance stability
        avg_time = sum(sample_times) / len(sample_times)
        min_time = min(sample_times)
        max_time = max(sample_times)
        
        # Performance should be stable
        stability_ratio = max_time / min_time
        assert stability_ratio < 5, f"Performance varies by {stability_ratio:.1f}x"
        
        # Should maintain realtime capability
        assert avg_time < 0.00002, f"Average {avg_time*1000:.3f}ms per sample"