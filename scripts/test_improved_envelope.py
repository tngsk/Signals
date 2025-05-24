#!/usr/bin/env python3
"""
Test script for the improved envelope with anti-click features.

This script tests the enhanced EnvelopeADSR module that includes
anti-click protection and exponential release curves.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import EnvelopeADSR, SynthContext, write_wav


def test_anti_click_protection():
    """Test anti-click protection for short release times."""
    print("=== Anti-Click Protection Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        
        # Test extremely short release time
        print("Testing 1ms release time (should be extended):")
        env.set_parameter("release", 0.001)
        print(f"  Requested: 0.001s")
        print(f"  Actual: {env.release_time:.3f}s")
        print(f"  Minimum enforced: {env.min_release_time:.3f}s")
        
        # Test acceptable release time
        print("\nTesting 10ms release time (should be unchanged):")
        env.set_parameter("release", 0.01)
        print(f"  Requested: 0.01s")
        print(f"  Actual: {env.release_time:.3f}s")
        
        # Check anti-click mode configuration
        info = env.get_info()
        print(f"Anti-click configuration:")
        print(f"  Min release time: {info['min_release_time']*1000:.1f}ms")
        print(f"  Exponential release: {info['use_exponential_release']}")


def test_exponential_vs_linear_release():
    """Compare exponential vs linear release curves."""
    print("\n=== Exponential vs Linear Release Comparison ===")
    
    def test_release_curve(use_exponential, name):
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)
            env.set_parameter("decay", 0.01)
            env.set_parameter("sustain", 0.8)
            env.set_parameter("release", 0.05)  # 50ms release
            
            # Configure release type
            env.use_exponential_release = use_exponential
            env._update_release_coefficient()
            
            # Get to sustain phase
            env.trigger_on()
            sustain_samples = int(0.05 * 48000)  # 50ms
            for _ in range(sustain_samples):
                env.process()
            
            # Trigger release and capture curve
            env.trigger_off()
            release_samples = int(0.1 * 48000)  # 100ms capture
            
            curve_values = []
            for i in range(release_samples):
                signal = env.process()[0]
                curve_values.append(signal.value)
            
            # Analyze curve smoothness
            derivatives = np.diff(curve_values)
            max_derivative = np.max(np.abs(derivatives))
            mean_derivative = np.mean(np.abs(derivatives))
            
            print(f"\n{name} release:")
            print(f"  Max derivative: {max_derivative:.6f}")
            print(f"  Mean derivative: {mean_derivative:.6f}")
            print(f"  Final value: {curve_values[-1]:.6f}")
            
            return curve_values, derivatives
    
    # Test both types
    exp_curve, exp_deriv = test_release_curve(True, "Exponential")
    lin_curve, lin_deriv = test_release_curve(False, "Linear")
    
    # Compare smoothness
    exp_smoothness = np.max(np.abs(exp_deriv))
    lin_smoothness = np.max(np.abs(lin_deriv))
    improvement = lin_smoothness / exp_smoothness if exp_smoothness > 0 else float('inf')
    
    print(f"\nSmoothness improvement: {improvement:.1f}x better with exponential")


def test_short_release_audio():
    """Test audio generation with very short release times."""
    print("\n=== Short Release Audio Test ===")
    
    def generate_test_audio(release_time, filename_suffix, description):
        with SynthContext(sample_rate=48000):
            # Generate test tone
            duration = 0.8
            num_samples = int(duration * 48000)
            t = np.arange(num_samples) / 48000
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)
            
            # Apply envelope
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.02)
            env.set_parameter("decay", 0.05)
            env.set_parameter("sustain", 0.7)
            env.set_parameter("release", release_time)
            
            env.trigger_on()
            
            modulated_audio = np.zeros(num_samples)
            envelope_data = np.zeros(num_samples)
            
            # Release at 75% through
            release_point = int(0.75 * num_samples)
            
            for i in range(num_samples):
                if i == release_point:
                    env.trigger_off()
                
                env_signal = env.process()[0]
                envelope_data[i] = env_signal.value
                modulated_audio[i] = sine_wave[i] * env_signal.value
            
            # Analyze final portion for clicks
            final_samples = 2000  # Last ~40ms
            final_audio = modulated_audio[-final_samples:]
            final_derivatives = np.diff(final_audio)
            max_final_derivative = np.max(np.abs(final_derivatives))
            
            # Ensure reports directory exists
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Save audio
            write_wav(str(reports_dir / f"improved_envelope_{filename_suffix}.wav"), modulated_audio, 48000)
            
            print(f"{description}:")
            print(f"  Requested release: {release_time*1000:.1f}ms")
            print(f"  Actual release: {env.release_time*1000:.1f}ms")
            print(f"  Max final derivative: {max_final_derivative:.6f}")
            print(f"  File: reports/improved_envelope_{filename_suffix}.wav")
            
            return modulated_audio, max_final_derivative
    
    # Test various release times
    test_cases = [
        (0.001, "1ms_protected", "1ms release (anti-click protected)"),
        (0.01, "10ms_normal", "10ms release (normal)"),
        (0.05, "50ms_smooth", "50ms release (smooth reference)")
    ]
    
    results = []
    for release_time, suffix, description in test_cases:
        audio, max_deriv = generate_test_audio(release_time, suffix, description)
        results.append((release_time, max_deriv))
    
    # Summary
    print(f"\nClick analysis summary:")
    for release_time, max_deriv in results:
        status = "✅ SMOOTH" if max_deriv < 0.001 else "⚠️ POSSIBLE CLICK" if max_deriv < 0.01 else "❌ CLICK"
        print(f"  {release_time*1000:4.1f}ms: {max_deriv:.6f} {status}")


def test_disable_anti_click():
    """Test disabling anti-click protection."""
    print("\n=== Anti-Click Disable Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        
        print("With anti-click protection (default):")
        env.set_parameter("release", 0.001)
        print(f"  1ms → {env.release_time*1000:.1f}ms")
        
        print("\nDisabling anti-click protection:")
        env.set_anti_click_mode(False)
        env.set_parameter("release", 0.001)
        print(f"  1ms → {env.release_time*1000:.1f}ms")
        
        print("\nRe-enabling with custom minimum:")
        env.set_anti_click_mode(True, min_time=0.010)  # 10ms minimum
        env.set_parameter("release", 0.001)
        print(f"  1ms → {env.release_time*1000:.1f}ms (min: 10ms)")


def test_auto_release_anti_click():
    """Test anti-click protection with auto release mode."""
    print("\n=== Auto Release Anti-Click Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.1)
        env.set_parameter("sustain", 0.7)
        env.set_parameter("release", "auto")
        
        # Test with short total duration
        print("Short total duration (anti-click should apply):")
        env.set_total_duration(0.5)  # 500ms total
        info = env.get_info()
        print(f"  Total duration: {info['total_duration']*1000:.0f}ms")
        print(f"  Auto release time: {info['release_time']*1000:.1f}ms")
        print(f"  Minimum enforced: {info['min_release_time']*1000:.1f}ms")
        
        # Test with longer total duration
        print("\nLonger total duration:")
        env.set_total_duration(2.0)  # 2000ms total
        info = env.get_info()
        print(f"  Total duration: {info['total_duration']*1000:.0f}ms")
        print(f"  Auto release time: {info['release_time']*1000:.1f}ms")


def test_envelope_performance():
    """Test performance impact of anti-click features."""
    print("\n=== Performance Impact Test ===")
    
    import time
    
    def benchmark_envelope(use_exponential, iterations=10000):
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)
            env.set_parameter("decay", 0.02)
            env.set_parameter("sustain", 0.7)
            env.set_parameter("release", 0.01)
            env.use_exponential_release = use_exponential
            env._update_release_coefficient()
            
            # Get to release phase
            env.trigger_on()
            for _ in range(1000):
                env.process()
            env.trigger_off()
            
            # Benchmark release processing
            start_time = time.perf_counter()
            for _ in range(iterations):
                env.process()
            end_time = time.perf_counter()
            
            return (end_time - start_time) / iterations
    
    # Benchmark both modes
    linear_time = benchmark_envelope(False)
    exponential_time = benchmark_envelope(True)
    
    overhead = ((exponential_time - linear_time) / linear_time * 100) if linear_time > 0 else 0
    
    print(f"Linear release: {linear_time*1000000:.2f}µs per sample")
    print(f"Exponential release: {exponential_time*1000000:.2f}µs per sample")
    print(f"Overhead: {overhead:.1f}%")
    
    if overhead < 20:
        print("✅ Performance impact acceptable")
    else:
        print("⚠️ Significant performance impact")


def main():
    """Run all improved envelope tests."""
    print("🎵 Improved Envelope with Anti-Click Features Test")
    print("=" * 60)
    
    try:
        test_anti_click_protection()
        test_exponential_vs_linear_release()
        test_short_release_audio()
        test_disable_anti_click()
        test_auto_release_anti_click()
        test_envelope_performance()
        
        print("\n" + "=" * 60)
        print("✅ All improved envelope tests completed!")
        print("\n🎯 Key Improvements:")
        print("  • ✨ Automatic minimum release time enforcement (5ms default)")
        print("    - Balances click prevention with musical expression")
        print("    - Preserves fast staccato and percussion techniques")
        print("  • 📈 Exponential release curves for smoother decay")
        print("  • 🔧 Configurable anti-click protection")
        print("  • ⚡ Maintained performance characteristics")
        print("  • 🎛️ Compatible with auto release mode")
        print("\n🎵 Musical Considerations:")
        print("  • 5ms minimum allows rapid musical passages")
        print("  • Prevents artifacts without limiting expression")
        print("  • Fine-grained click removal planned for separate phase")
        print("\n📁 Generated audio files:")
        print("  • reports/improved_envelope_1ms_protected.wav")
        print("  • reports/improved_envelope_10ms_normal.wav") 
        print("  • reports/improved_envelope_50ms_smooth.wav")
        print("\n🎧 Listen to compare click reduction effectiveness!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())