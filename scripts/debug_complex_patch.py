#!/usr/bin/env python3
"""
Debug script for complex patch rendering issues.

This script isolates and tests the complex patch functionality
to identify why Step 16 causes the demo to hang.
"""

import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from signals import SynthEngine, Patch


def create_complex_patch_file():
    """Create the complex patch file for testing."""
    complex_patch = """name: "Complex Graph Demo"
description: "Multi-module patch to demonstrate graph features"

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
    
    with open("debug_complex.yaml", "w") as f:
        f.write(complex_patch)
    print("✅ Created debug_complex.yaml")


def test_patch_loading():
    """Test patch loading step by step."""
    print("\n🔍 Testing patch loading...")
    
    try:
        patch = Patch.from_file("debug_complex.yaml")
        print(f"✅ Patch loaded: {patch.name}")
        print(f"   Modules: {len(patch.modules)}")
        print(f"   Connections: {len(patch.connections)}")
        print(f"   Sequence: {len(patch.sequence)}")
        return patch
    except Exception as e:
        print(f"❌ Patch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_engine_creation():
    """Test engine creation."""
    print("\n🔍 Testing engine creation...")
    
    try:
        engine = SynthEngine(sample_rate=48000)
        print(f"✅ Engine created: {engine.sample_rate} Hz")
        return engine
    except Exception as e:
        print(f"❌ Engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_patch_in_engine(engine):
    """Test loading patch into engine."""
    print("\n🔍 Testing patch loading into engine...")
    
    try:
        patch = engine.load_patch("debug_complex.yaml")
        print(f"✅ Patch loaded into engine: {patch.name}")
        
        info = engine.get_patch_info()
        print(f"   Modules: {info.get('modules', [])}")
        print(f"   Execution order: {info.get('execution_order', [])}")
        print(f"   Duration: {info.get('duration', 0.0)}s")
        
        return True
    except Exception as e:
        print(f"❌ Engine patch loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_render(engine, duration=0.1):
    """Test very short audio rendering."""
    print(f"\n🔍 Testing short render ({duration}s)...")
    
    try:
        start_time = time.time()
        audio = engine.render(duration=duration)
        end_time = time.time()
        
        print(f"✅ Short render successful:")
        print(f"   Duration: {duration}s")
        print(f"   Samples: {len(audio)}")
        print(f"   Render time: {end_time - start_time:.3f}s")
        
        # Check audio content
        if len(audio) > 0:
            import numpy as np
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio ** 2))
            print(f"   Peak: {peak:.6f}")
            print(f"   RMS: {rms:.6f}")
            
            # Check for silence (potential infinite loop indicator)
            if peak == 0.0:
                print("⚠️  Warning: Audio is completely silent!")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Short render failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_graph_directly():
    """Test module graph creation directly."""
    print("\n🔍 Testing module graph creation...")
    
    try:
        from signals.graph import ModuleGraph
        
        patch = Patch.from_file("debug_complex.yaml")
        print("✅ Patch loaded for graph test")
        
        graph = ModuleGraph(patch)
        print("✅ Module graph created")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Execution order: {graph.execution_order}")
        
        # Test single sample processing
        print("\n🔍 Testing single sample processing...")
        outputs = graph.process_sample()
        print(f"✅ Single sample processed")
        print(f"   Output modules: {list(outputs.keys())}")
        
        for module_id, signals in outputs.items():
            print(f"   {module_id}: {len(signals)} signals")
            for i, signal in enumerate(signals):
                print(f"     Signal {i}: type={signal.type.value}, value={signal.value}")
        
        return True
    except Exception as e:
        print(f"❌ Module graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_render_with_timeout(engine, duration=0.5, timeout=5.0):
    """Test render with timeout to detect hanging."""
    print(f"\n🔍 Testing render with timeout ({timeout}s timeout)...")
    
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Render timed out!")
    
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        start_time = time.time()
        audio = engine.render(duration=duration)
        end_time = time.time()
        
        # Cancel timeout
        signal.alarm(0)
        
        print(f"✅ Timed render successful:")
        print(f"   Duration: {duration}s")
        print(f"   Samples: {len(audio)}")
        print(f"   Render time: {end_time - start_time:.3f}s")
        
        return True
    except TimeoutError:
        print(f"❌ Render timed out after {timeout}s - likely infinite loop!")
        return False
    except Exception as e:
        print(f"❌ Timed render failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        signal.alarm(0)  # Make sure to cancel timeout


def main():
    """Run all debug tests."""
    print("🐛 Debug: Complex Patch Rendering Issue")
    print("=" * 50)
    
    # Step 1: Create patch file
    create_complex_patch_file()
    
    # Step 2: Test patch loading
    patch = test_patch_loading()
    if not patch:
        return
    
    # Step 3: Test engine creation
    engine = test_engine_creation()
    if not engine:
        return
    
    # Step 4: Test patch in engine
    if not test_patch_in_engine(engine):
        return
    
    # Step 5: Test module graph directly
    if not test_module_graph_directly():
        return
    
    # Step 6: Test very short render
    if not test_short_render(engine, duration=0.01):
        return
    
    # Step 7: Test slightly longer render
    if not test_short_render(engine, duration=0.1):
        return
    
    # Step 8: Test with timeout to detect hanging
    if not test_render_with_timeout(engine, duration=0.5, timeout=5.0):
        print("\n💡 Problem identified: Render process hangs!")
        print("   This suggests an infinite loop in the module graph processing.")
        return
    
    # Step 9: If we get here, try the full duration
    print("\n🔍 Testing full duration render...")
    if test_short_render(engine, duration=1.0):
        print("\n✅ All tests passed! Complex patch works correctly.")
    else:
        print("\n❌ Full duration render failed.")
    
    # Cleanup
    try:
        Path("debug_complex.yaml").unlink()
        print("\n🗑️  Cleaned up debug_complex.yaml")
    except:
        pass


if __name__ == "__main__":
    main()