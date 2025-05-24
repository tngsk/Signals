#!/usr/bin/env python3
"""
Debug script for module performance analysis.

This script measures the processing time of individual modules
to identify performance bottlenecks in the synthesis pipeline.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from signals import Oscillator, EnvelopeADSR, Mixer, Signal, SignalType


def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def test_oscillator_performance():
    """Test oscillator processing performance."""
    print("\n🔍 Testing Oscillator Performance")
    print("-" * 40)
    
    sample_rate = 48000
    osc = Oscillator(sample_rate)
    osc.set_parameter("frequency", 440.0)
    osc.set_parameter("waveform", "sine")
    
    # Test single sample processing
    _, single_time = time_function(osc.process)
    print(f"Single sample: {single_time*1000:.3f}ms")
    
    # Test multiple samples
    num_samples = 1000
    start_time = time.perf_counter()
    for _ in range(num_samples):
        osc.process()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    print(f"1000 samples: {total_time:.3f}s total, {avg_time*1000:.3f}ms avg")
    print(f"Samples per second: {num_samples/total_time:.0f}")
    
    # Test different waveforms
    waveforms = ["sine", "square", "triangle", "saw"]
    for waveform in waveforms:
        osc.set_parameter("waveform", waveform)
        _, waveform_time = time_function(osc.process)
        print(f"{waveform:8s}: {waveform_time*1000:.3f}ms")


def test_envelope_performance():
    """Test envelope processing performance."""
    print("\n🔍 Testing Envelope Performance")
    print("-" * 40)
    
    sample_rate = 48000
    env = EnvelopeADSR(sample_rate)
    env.set_parameter("attack", 0.1)
    env.set_parameter("decay", 0.5)
    env.set_parameter("sustain", 0.3)
    env.set_parameter("release", 0.3)
    
    # Trigger envelope
    env.trigger_on()
    
    # Test single sample processing
    _, single_time = time_function(env.process)
    print(f"Single sample: {single_time*1000:.3f}ms")
    
    # Test multiple samples
    num_samples = 1000
    start_time = time.perf_counter()
    for _ in range(num_samples):
        env.process()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    print(f"1000 samples: {total_time:.3f}s total, {avg_time*1000:.3f}ms avg")
    print(f"Samples per second: {num_samples/total_time:.0f}")


def test_mixer_performance():
    """Test mixer processing performance."""
    print("\n🔍 Testing Mixer Performance")
    print("-" * 40)
    
    mixer = Mixer(num_inputs=2)
    mixer.set_parameter("gain1", 0.7)
    mixer.set_parameter("gain2", 0.3)
    
    # Create test signals
    signal1 = Signal(SignalType.AUDIO, 0.5)
    signal2 = Signal(SignalType.AUDIO, 0.3)
    inputs = [signal1, signal2]
    
    # Test single sample processing
    _, single_time = time_function(mixer.process, inputs)
    print(f"Single sample: {single_time*1000:.3f}ms")
    
    # Test multiple samples
    num_samples = 1000
    start_time = time.perf_counter()
    for _ in range(num_samples):
        mixer.process(inputs)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    print(f"1000 samples: {total_time:.3f}s total, {avg_time*1000:.3f}ms avg")
    print(f"Samples per second: {num_samples/total_time:.0f}")


def test_full_chain_performance():
    """Test full processing chain performance."""
    print("\n🔍 Testing Full Chain Performance")
    print("-" * 40)
    
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
    
    # Test single full chain processing
    def process_chain():
        osc1_out = osc1.process()
        osc2_out = osc2.process()
        mixer_out = mixer.process([osc1_out[0], osc2_out[0]])
        env_out = env.process([mixer_out[0]])
        return env_out
    
    _, single_time = time_function(process_chain)
    print(f"Single chain: {single_time*1000:.3f}ms")
    
    # Test multiple chain processings
    num_samples = 1000
    start_time = time.perf_counter()
    for _ in range(num_samples):
        process_chain()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    print(f"1000 chains: {total_time:.3f}s total, {avg_time*1000:.3f}ms avg")
    print(f"Chains per second: {num_samples/total_time:.0f}")
    
    # Calculate expected time for 1 second of audio
    samples_per_second = sample_rate
    expected_time = avg_time * samples_per_second
    print(f"Expected time for 1s audio: {expected_time:.3f}s (realtime factor: {expected_time:.1f}x)")


def benchmark_graph_vs_manual():
    """Compare ModuleGraph performance vs manual processing."""
    print("\n🔍 Comparing Graph vs Manual Processing")
    print("-" * 40)
    
    from signals import SynthEngine, Patch
    
    # Create patch file
    patch_content = """name: "Performance Test"
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
"""
    
    with open("perf_test.yaml", "w") as f:
        f.write(patch_content)
    
    try:
        # Test ModuleGraph approach
        engine = SynthEngine(sample_rate=48000)
        patch = engine.load_patch("perf_test.yaml")
        
        print("Testing ModuleGraph processing...")
        start_time = time.perf_counter()
        
        # Process just 100 samples to avoid timeout
        num_samples = 100
        for _ in range(num_samples):
            _ = engine.current_graph.process_sample()
        
        end_time = time.perf_counter()
        graph_time = end_time - start_time
        graph_avg = graph_time / num_samples
        
        print(f"ModuleGraph: {graph_time:.3f}s total, {graph_avg*1000:.3f}ms avg")
        print(f"Samples per second: {num_samples/graph_time:.0f}")
        
    except Exception as e:
        print(f"ModuleGraph test failed: {e}")
    finally:
        try:
            Path("perf_test.yaml").unlink()
        except:
            pass


def main():
    """Run all performance tests."""
    print("🚀 Module Performance Analysis")
    print("=" * 50)
    
    test_oscillator_performance()
    test_envelope_performance() 
    test_mixer_performance()
    test_full_chain_performance()
    benchmark_graph_vs_manual()
    
    print("\n📊 Analysis Summary:")
    print("If any module takes >1ms per sample, that's a problem.")
    print("For real-time audio (48kHz), each sample must process in <20μs.")
    print("Current performance issues likely in ModuleGraph processing logic.")


if __name__ == "__main__":
    main()