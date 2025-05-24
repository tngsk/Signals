#!/usr/bin/env python3
"""
Detailed profiling script for ModuleGraph performance issues.

This script performs line-by-line profiling of the graph processing
to identify exactly where the performance bottleneck occurs.
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from signals import SynthEngine, Patch
from signals.graph import ModuleGraph


def create_test_patch():
    """Create a test patch file."""
    patch_content = """name: "Performance Debug"
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
    
    with open("debug_perf.yaml", "w") as f:
        f.write(patch_content)


def profile_graph_creation():
    """Profile the graph creation process."""
    print("🔍 Profiling graph creation...")
    
    patch = Patch.from_file("debug_perf.yaml")
    
    start_time = time.perf_counter()
    graph = ModuleGraph(patch)
    end_time = time.perf_counter()
    
    print(f"Graph creation time: {end_time - start_time:.6f}s")
    return graph


def profile_single_sample():
    """Profile a single sample processing."""
    print("\n🔍 Profiling single sample processing...")
    
    patch = Patch.from_file("debug_perf.yaml")
    graph = ModuleGraph(patch)
    
    # Warm up
    graph.process_sample()
    
    # Profile the actual call
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.perf_counter()
    result = graph.process_sample()
    end_time = time.perf_counter()
    
    profiler.disable()
    
    print(f"Single sample time: {end_time - start_time:.6f}s")
    
    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop functions by cumulative time:")
    stats.print_stats(10)
    
    return result


def trace_sample_processing():
    """Trace sample processing step by step."""
    print("\n🔍 Tracing sample processing step by step...")
    
    patch = Patch.from_file("debug_perf.yaml")
    graph = ModuleGraph(patch)
    
    print(f"Execution order: {graph.execution_order}")
    print(f"Node count: {len(graph.nodes)}")
    
    # Reset nodes
    for node in graph.nodes.values():
        node.reset_cycle()
    
    # Process each module individually and time it
    outputs = {}
    total_time = 0
    
    for module_id in graph.execution_order:
        start_time = time.perf_counter()
        
        node = graph.nodes[module_id]
        print(f"\nProcessing {module_id} ({node.module.__class__.__name__}):")
        print(f"  Input count: {node.module.input_count}")
        print(f"  Input connections: {node.input_connections}")
        
        # Gather inputs manually
        inputs = []
        for input_idx in range(node.module.input_count):
            if input_idx in node.input_connections:
                source_id, output_idx = node.input_connections[input_idx]
                print(f"    Input {input_idx}: from {source_id}.{output_idx}")
                
                if source_id in outputs:
                    source_outputs = outputs[source_id]
                    if output_idx < len(source_outputs):
                        inputs.append(source_outputs[output_idx])
                        print(f"      Value: {source_outputs[output_idx].value}")
                    else:
                        from signals.module import Signal, SignalType
                        inputs.append(Signal(SignalType.AUDIO, 0.0))
                        print(f"      Default (out of range)")
                else:
                    from signals.module import Signal, SignalType
                    inputs.append(Signal(SignalType.AUDIO, 0.0))
                    print(f"      Default (source not ready)")
            else:
                from signals.module import Signal, SignalType
                inputs.append(Signal(SignalType.AUDIO, 0.0))
                print(f"    Input {input_idx}: default (unconnected)")
        
        # Process module
        module_start = time.perf_counter()
        if inputs or node.module.input_count == 0:
            node_outputs = node.module.process(inputs if inputs else None)
        else:
            node_outputs = node.module.process()
        module_end = time.perf_counter()
        
        outputs[module_id] = node_outputs
        node.cached_outputs = node_outputs
        node.processed_this_cycle = True
        
        end_time = time.perf_counter()
        
        module_time = module_end - module_start
        total_module_time = end_time - start_time
        overhead_time = total_module_time - module_time
        
        print(f"  Module processing: {module_time*1000:.3f}ms")
        print(f"  Total time: {total_module_time*1000:.3f}ms")
        print(f"  Overhead: {overhead_time*1000:.3f}ms ({overhead_time/total_module_time*100:.1f}%)")
        print(f"  Outputs: {len(node_outputs)} signals")
        
        total_time += total_module_time
    
    print(f"\nTotal processing time: {total_time*1000:.3f}ms")
    
    return outputs


def compare_implementations():
    """Compare different implementation approaches."""
    print("\n🔍 Comparing implementation approaches...")
    
    patch = Patch.from_file("debug_perf.yaml")
    
    # Test 1: Current ModuleGraph implementation
    print("\n1. Current ModuleGraph:")
    graph = ModuleGraph(patch)
    
    start_time = time.perf_counter()
    for _ in range(10):
        graph.process_sample()
    end_time = time.perf_counter()
    
    current_time = (end_time - start_time) / 10
    print(f"   Average time: {current_time*1000:.3f}ms")
    
    # Test 2: Direct module processing
    print("\n2. Direct module processing:")
    from signals import Oscillator, Mixer, EnvelopeADSR
    
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
    
    def process_direct():
        osc1_out = osc1.process()
        osc2_out = osc2.process()
        mixer_out = mixer.process([osc1_out[0], osc2_out[0]])
        env_out = env.process([mixer_out[0]])
        return env_out
    
    start_time = time.perf_counter()
    for _ in range(10):
        process_direct()
    end_time = time.perf_counter()
    
    direct_time = (end_time - start_time) / 10
    print(f"   Average time: {direct_time*1000:.3f}ms")
    print(f"   Speedup: {current_time/direct_time:.1f}x faster")


def analyze_memory_usage():
    """Analyze memory usage patterns."""
    print("\n🔍 Analyzing memory usage...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    patch = Patch.from_file("debug_perf.yaml")
    print(f"After patch load: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    graph = ModuleGraph(patch)
    print(f"After graph creation: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Process many samples
    for i in range(1000):
        graph.process_sample()
        if i % 100 == 0:
            print(f"After {i} samples: {process.memory_info().rss / 1024 / 1024:.1f} MB")


def main():
    """Run detailed profiling analysis."""
    print("🔬 Detailed ModuleGraph Performance Analysis")
    print("=" * 60)
    
    create_test_patch()
    
    try:
        profile_graph_creation()
        profile_single_sample()
        trace_sample_processing()
        compare_implementations()
        analyze_memory_usage()
        
        print("\n📊 Summary:")
        print("- Check the profiling output for hotspots")
        print("- Look for unexpected function calls or loops")
        print("- Compare graph vs direct processing times")
        print("- Monitor memory usage for leaks")
        
    finally:
        try:
            Path("debug_perf.yaml").unlink()
        except:
            pass


if __name__ == "__main__":
    main()