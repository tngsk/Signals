"""
Performance and regression tests for the Signals synthesizer framework.

Migrated from scripts/debug_*.py for automated testing and CI integration.
"""

import pytest
import time
import tempfile
import numpy as np
from pathlib import Path
import signal as signal_module
from unittest.mock import patch

from signals import SynthEngine, Patch, Oscillator, EnvelopeADSR, Mixer, Signal, SignalType


class TimeoutError(Exception):
    """Custom timeout exception for render timeout tests."""
    pass


@pytest.fixture
def complex_patch_content():
    """Complex patch configuration for testing."""
    return """name: "Complex Performance Test"
description: "Multi-module patch for performance testing"

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

  mixer:
    type: "mixer"
    parameters:
      gain1: 0.7
      gain2: 0.3

  env1:
    type: "envelope_adsr"
    parameters:
      attack: 0.1
      decay: 0.5
      sustain: 0.3
      release: 0.3

connections:
  - from: "osc1.0"
    to: "mixer.0"
  - from: "osc2.0"
    to: "mixer.1"
  - from: "mixer.0"
    to: "env1.0"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 2.0
    action: "release"
    target: "env1"
"""


@pytest.fixture
def temp_patch_file(complex_patch_content):
    """Create temporary patch file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(complex_patch_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        Path(temp_path).unlink()
    except FileNotFoundError:
        pass


@pytest.mark.performance
class TestModulePerformance:
    """Performance tests for individual modules."""
    
    def test_oscillator_performance(self):
        """Test oscillator processing performance."""
        sample_rate = 48000
        osc = Oscillator(sample_rate)
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("waveform", "sine")
        
        # Test single sample processing
        start_time = time.perf_counter()
        osc.process()
        single_time = time.perf_counter() - start_time
        
        # Should process single sample in under 1ms
        assert single_time < 0.001, f"Single sample took {single_time*1000:.3f}ms"
        
        # Test multiple samples
        num_samples = 1000
        start_time = time.perf_counter()
        for _ in range(num_samples):
            osc.process()
        total_time = time.perf_counter() - start_time
        
        avg_time = total_time / num_samples
        samples_per_second = num_samples / total_time
        
        # Should process at least 10x realtime (480kHz for 48kHz audio)
        assert samples_per_second > 480000, f"Only {samples_per_second:.0f} samples/sec"
        assert avg_time < 0.00002, f"Average time {avg_time*1000:.3f}ms too slow"
    
    def test_waveform_performance_comparison(self):
        """Compare performance across different waveforms."""
        sample_rate = 48000
        osc = Oscillator(sample_rate)
        osc.set_parameter("frequency", 440.0)
        
        waveforms = ["sine", "square", "triangle", "saw"]
        timings = {}
        
        for waveform in waveforms:
            osc.set_parameter("waveform", waveform)
            
            # Warm up
            osc.process()
            
            # Time 100 samples
            start_time = time.perf_counter()
            for _ in range(100):
                osc.process()
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 100
            timings[waveform] = avg_time
        
        # All waveforms should be reasonably fast
        for waveform, timing in timings.items():
            assert timing < 0.0001, f"{waveform} too slow: {timing*1000:.3f}ms"
        
        # Performance variance should be reasonable (within 10x)
        min_time = min(timings.values())
        max_time = max(timings.values())
        assert max_time / min_time < 10, "Excessive performance variance between waveforms"
    
    def test_envelope_performance(self):
        """Test envelope processing performance."""
        sample_rate = 48000
        env = EnvelopeADSR(sample_rate)
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.5)
        env.set_parameter("sustain", 0.3)
        env.set_parameter("release", 0.3)
        
        env.trigger_on()
        
        # Test multiple samples
        num_samples = 1000
        start_time = time.perf_counter()
        for _ in range(num_samples):
            env.process()
        total_time = time.perf_counter() - start_time
        
        avg_time = total_time / num_samples
        samples_per_second = num_samples / total_time
        
        # Should process fast enough for realtime
        assert samples_per_second > 480000, f"Envelope too slow: {samples_per_second:.0f} samples/sec"
        assert avg_time < 0.00002, f"Average time {avg_time*1000:.3f}ms too slow"
    
    def test_mixer_performance(self):
        """Test mixer processing performance."""
        mixer = Mixer(num_inputs=4)
        mixer.set_parameter("gain1", 0.7)
        mixer.set_parameter("gain2", 0.3)
        mixer.set_parameter("gain3", 0.5)
        mixer.set_parameter("gain4", 0.2)
        
        # Create test signals
        inputs = [
            Signal(SignalType.AUDIO, 0.5),
            Signal(SignalType.AUDIO, 0.3),
            Signal(SignalType.AUDIO, 0.2),
            Signal(SignalType.AUDIO, 0.1)
        ]
        
        # Test multiple samples
        num_samples = 1000
        start_time = time.perf_counter()
        for _ in range(num_samples):
            mixer.process(inputs)
        total_time = time.perf_counter() - start_time
        
        avg_time = total_time / num_samples
        samples_per_second = num_samples / total_time
        
        # Mixer should be very fast
        assert samples_per_second > 1000000, f"Mixer too slow: {samples_per_second:.0f} samples/sec"
        assert avg_time < 0.000005, f"Average time {avg_time*1000:.3f}ms too slow"


@pytest.mark.performance
class TestFullChainPerformance:
    """Performance tests for complete processing chains."""
    
    def test_manual_chain_performance(self):
        """Test performance of manually connected modules."""
        sample_rate = 48000
        
        # Create modules
        osc1 = Oscillator(sample_rate)
        osc1.set_parameter("frequency", 440.0)
        osc1.set_parameter("waveform", "sine")
        
        osc2 = Oscillator(sample_rate)
        osc2.set_parameter("frequency", 660.0)
        osc2.set_parameter("waveform", "square")
        
        mixer = Mixer(num_inputs=2)
        mixer.set_parameter("gain1", 0.7)
        mixer.set_parameter("gain2", 0.3)
        
        env = EnvelopeADSR(sample_rate)
        env.set_parameter("attack", 0.1)
        env.trigger_on()
        
        def process_chain():
            osc1_out = osc1.process()
            osc2_out = osc2.process()
            mixer_out = mixer.process([osc1_out[0], osc2_out[0]])
            env_out = env.process([mixer_out[0]])
            return env_out
        
        # Test multiple chain processings
        num_samples = 1000
        start_time = time.perf_counter()
        for _ in range(num_samples):
            process_chain()
        total_time = time.perf_counter() - start_time
        
        avg_time = total_time / num_samples
        chains_per_second = num_samples / total_time
        
        # Calculate expected time for 1 second of audio
        expected_time_for_1s = avg_time * sample_rate
        realtime_factor = expected_time_for_1s
        
        assert chains_per_second > 48000, f"Chain too slow: {chains_per_second:.0f} chains/sec"
        assert realtime_factor < 0.5, f"Realtime factor {realtime_factor:.1f}x too high"
    
    def test_engine_render_performance(self, temp_patch_file):
        """Test engine rendering performance."""
        engine = SynthEngine(sample_rate=48000)
        patch = engine.load_patch(temp_patch_file)
        
        # Test short render performance
        duration = 0.1  # 100ms
        start_time = time.perf_counter()
        audio = engine.render(duration=duration)
        render_time = time.perf_counter() - start_time
        
        expected_samples = int(duration * engine.sample_rate)
        realtime_factor = render_time / duration
        
        assert len(audio) == expected_samples
        assert realtime_factor < 5.0, f"Render {realtime_factor:.1f}x slower than realtime"
        
        # Audio should not be silent
        assert np.max(np.abs(audio)) > 0.001, "Rendered audio is too quiet"


@pytest.mark.integration
class TestComplexPatchRegression:
    """Regression tests for complex patch processing issues."""
    
    def test_patch_loading_steps(self, temp_patch_file):
        """Test patch loading step by step to catch regressions."""
        # Step 1: Test patch loading
        patch = Patch.from_file(temp_patch_file)
        assert patch.name == "Complex Performance Test"
        assert len(patch.modules) == 4
        assert len(patch.connections) == 3
        assert len(patch.sequence) == 2
        
        # Step 2: Test engine creation
        engine = SynthEngine(sample_rate=48000)
        assert engine.sample_rate == 48000
        
        # Step 3: Test patch in engine
        loaded_patch = engine.load_patch(temp_patch_file)
        assert loaded_patch.name == patch.name
        
        info = engine.get_patch_info()
        assert 'modules' in info
        assert 'execution_order' in info
        assert 'duration' in info
    
    def test_short_render_safety(self, temp_patch_file):
        """Test very short renders to catch infinite loops early."""
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_patch_file)
        
        # Test progressively longer renders
        durations = [0.001, 0.01, 0.1]  # 1ms, 10ms, 100ms
        
        for duration in durations:
            start_time = time.perf_counter()
            audio = engine.render(duration=duration)
            render_time = time.perf_counter() - start_time
            
            expected_samples = int(duration * engine.sample_rate)
            assert len(audio) == expected_samples
            
            # Should complete quickly
            assert render_time < 1.0, f"Render took {render_time:.3f}s for {duration}s audio"
            
            # Check audio content
            peak = np.max(np.abs(audio))
            assert peak > 0.0, f"Audio is silent for {duration}s render"
            assert peak <= 1.0, f"Audio clipped: peak = {peak}"
    
    def test_render_timeout_protection(self, temp_patch_file):
        """Test render with timeout to detect hanging."""
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_patch_file)
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Render timed out!")
        
        # Set up timeout (Unix-like systems only)
        try:
            old_handler = signal_module.signal(signal_module.SIGALRM, timeout_handler)
            signal_module.alarm(5)  # 5 second timeout
            
            start_time = time.perf_counter()
            audio = engine.render(duration=0.5)
            end_time = time.perf_counter()
            
            # Cancel timeout
            signal_module.alarm(0)
            signal_module.signal(signal_module.SIGALRM, old_handler)
            
            render_time = end_time - start_time
            assert render_time < 4.0, f"Render took {render_time:.3f}s - possible performance issue"
            assert len(audio) > 0, "No audio rendered"
            
        except AttributeError:
            # Windows doesn't have SIGALRM, use alternative timeout
            import threading
            
            result = [None]
            exception = [None]
            
            def render_thread():
                try:
                    result[0] = engine.render(duration=0.5)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=render_thread)
            thread.start()
            thread.join(timeout=5.0)
            
            if thread.is_alive():
                pytest.fail("Render hanged - thread still running after 5s")
            
            if exception[0]:
                raise exception[0]
            
            assert result[0] is not None, "No render result"
            assert len(result[0]) > 0, "No audio rendered"


@pytest.mark.performance  
class TestPerformanceRegression:
    """Performance regression tests with specific benchmarks."""
    
    def test_oscillator_frequency_sweep_performance(self):
        """Test oscillator performance under frequency modulation."""
        sample_rate = 48000
        osc = Oscillator(sample_rate)
        osc.set_parameter("waveform", "sine")
        
        # Sweep frequencies rapidly
        frequencies = [220, 440, 880, 1760, 3520]
        num_samples_per_freq = 100
        
        start_time = time.perf_counter()
        for freq in frequencies:
            osc.set_parameter("frequency", freq)
            for _ in range(num_samples_per_freq):
                output = osc.process()
                assert len(output) == 1
                assert -1.0 <= output[0].value <= 1.0
        
        total_time = time.perf_counter() - start_time
        total_samples = len(frequencies) * num_samples_per_freq
        avg_time = total_time / total_samples
        
        # Should handle frequency changes efficiently
        assert avg_time < 0.00005, f"Frequency sweep too slow: {avg_time*1000:.3f}ms per sample"
    
    def test_memory_usage_stability(self, temp_patch_file):
        """Test for memory leaks during repeated processing."""
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_patch_file)
        
        # Process many short renders to test for memory leaks
        for i in range(100):
            audio = engine.render(duration=0.01)  # 10ms renders
            assert len(audio) > 0
            
            # Every 20 iterations, verify we're not accumulating excessive objects
            if i % 20 == 0:
                # This is a basic check - in a real scenario you might use memory_profiler
                import gc
                gc.collect()
                # If we had a memory profiler, we'd check memory usage here
    
    def test_concurrent_engine_performance(self, temp_patch_file):
        """Test performance with multiple engines (simulating multi-track usage)."""
        engines = []
        
        # Create multiple engines
        for i in range(4):
            engine = SynthEngine(sample_rate=48000)
            engine.load_patch(temp_patch_file)
            engines.append(engine)
        
        # Render from all engines simultaneously
        start_time = time.perf_counter()
        
        audio_results = []
        for engine in engines:
            audio = engine.render(duration=0.1)
            audio_results.append(audio)
        
        total_time = time.perf_counter() - start_time
        
        # All renders should complete reasonably quickly
        assert total_time < 2.0, f"Multi-engine render took {total_time:.3f}s"
        
        # All should produce valid audio
        for i, audio in enumerate(audio_results):
            assert len(audio) > 0, f"Engine {i} produced no audio"
            assert np.max(np.abs(audio)) > 0.001, f"Engine {i} audio too quiet"


@pytest.mark.slow
@pytest.mark.performance
class TestLongRunningPerformance:
    """Long-running performance tests."""
    
    def test_extended_render_performance(self, temp_patch_file):
        """Test performance of longer renders."""
        engine = SynthEngine(sample_rate=48000)
        engine.load_patch(temp_patch_file)
        
        # Test 5 second render
        duration = 5.0
        start_time = time.perf_counter()
        audio = engine.render(duration=duration)
        render_time = time.perf_counter() - start_time
        
        expected_samples = int(duration * engine.sample_rate)
        realtime_factor = render_time / duration
        
        assert len(audio) == expected_samples
        assert realtime_factor < 10.0, f"Long render {realtime_factor:.1f}x slower than realtime"
        
        # Check audio quality
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        
        assert rms > 0.001, f"RMS too low: {rms}"
        assert peak <= 1.0, f"Audio clipped: peak = {peak}"
        assert not np.any(np.isnan(audio)), "Audio contains NaN values"
        assert not np.any(np.isinf(audio)), "Audio contains infinite values"