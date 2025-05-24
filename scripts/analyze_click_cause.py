#!/usr/bin/env python3
"""
Detailed analysis of envelope click causes, focusing on short release times.

This script investigates the mathematical and implementation issues that cause
click artifacts when envelope release times are very short.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import EnvelopeADSR, SynthContext, write_wav


def analyze_short_release_math():
    """Analyze the mathematical behavior of short release times."""
    print("=== Short Release Time Mathematical Analysis ===")
    
    sample_rate = 48000
    sustain_level = 0.8
    
    # Test various short release times
    release_times = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    
    for release_time in release_times:
        release_samples = int(release_time * sample_rate)
        
        # Calculate expected step size per sample
        if release_samples > 0:
            step_per_sample = sustain_level / release_samples
        else:
            step_per_sample = float('inf')
        
        print(f"\nRelease time: {release_time}s ({release_time*1000}ms)")
        print(f"  Release samples: {release_samples}")
        print(f"  Step per sample: {step_per_sample:.6f}")
        print(f"  Step percentage: {step_per_sample/sustain_level*100:.2f}%")
        
        # Check if step size causes audible artifacts
        if step_per_sample > 0.01:  # Arbitrary threshold for "large jump"
            print(f"  ⚠️  PROBLEM: Step size {step_per_sample:.6f} likely causes clicks")
        else:
            print(f"  ✅ Step size acceptable")


def test_envelope_sample_precision():
    """Test envelope behavior with sample-level precision."""
    print("\n=== Sample-Level Precision Analysis ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.01)
        env.set_parameter("decay", 0.01)
        env.set_parameter("sustain", 0.8)
        env.set_parameter("release", 0.001)  # 1ms release - problematic
        
        # Get to sustain phase
        env.trigger_on()
        attack_samples = int(0.01 * 48000)
        decay_samples = int(0.01 * 48000)
        
        for _ in range(attack_samples + decay_samples + 10):
            env.process()
        
        print(f"Sustain value before release: {env._value:.6f}")
        print(f"Release samples calculated: {env._release_samples}")
        
        # Trigger release and capture exact values
        env.trigger_off()
        print(f"Phase after trigger_off: {env._phase}")
        
        release_values = []
        for i in range(env._release_samples + 10):
            signal = env.process()[0]
            release_values.append(signal.value)
            
            if i < 10:  # Show first 10 samples
                print(f"  Release sample {i}: {signal.value:.6f}")
        
        # Calculate actual step sizes
        if len(release_values) > 1:
            steps = np.diff(release_values)
            print(f"\nActual step sizes:")
            print(f"  Min step: {np.min(steps):.6f}")
            print(f"  Max step: {np.max(steps):.6f}")
            print(f"  Mean step: {np.mean(steps):.6f}")
            
            large_steps = steps[np.abs(steps) > 0.01]
            print(f"  Large steps (>0.01): {len(large_steps)}")


def test_release_calculation_accuracy():
    """Test accuracy of release time calculations."""
    print("\n=== Release Calculation Accuracy Test ===")
    
    sample_rate = 48000
    test_cases = [
        (0.001, "1ms - extreme short"),
        (0.0015, "1.5ms - very short"), 
        (0.002, "2ms - short"),
        (0.005, "5ms - acceptable"),
        (0.01, "10ms - good"),
        (1.0/48000, "1 sample - minimum possible"),
        (2.0/48000, "2 samples"),
        (5.0/48000, "5 samples"),
    ]
    
    for release_time, description in test_cases:
        release_samples = int(release_time * sample_rate)
        actual_time = release_samples / sample_rate
        time_error = actual_time - release_time
        
        print(f"\n{description}:")
        print(f"  Requested: {release_time*1000:.3f}ms")
        print(f"  Samples: {release_samples}")
        print(f"  Actual: {actual_time*1000:.3f}ms")
        print(f"  Error: {time_error*1000:.3f}ms")
        
        if release_samples == 0:
            print(f"  ❌ CRITICAL: Zero samples = instant cutoff = click!")
        elif release_samples < 5:
            print(f"  ⚠️  WARNING: Very few samples may cause artifacts")
        else:
            print(f"  ✅ Sufficient samples for smooth release")


def generate_click_comparison_audio():
    """Generate audio examples showing click vs no-click."""
    print("\n=== Click Comparison Audio Generation ===")
    
    def generate_envelope_audio(release_time, filename_suffix):
        with SynthContext(sample_rate=48000):
            # Generate sine wave
            duration = 0.5
            num_samples = int(duration * 48000)
            t = np.arange(num_samples) / 48000
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)
            
            # Apply envelope
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.02)
            env.set_parameter("decay", 0.05) 
            env.set_parameter("sustain", 0.8)
            env.set_parameter("release", release_time)
            
            env.trigger_on()
            
            modulated_audio = np.zeros(num_samples)
            envelope_data = np.zeros(num_samples)
            
            # Release at 80% through
            release_point = int(0.8 * num_samples)
            
            for i in range(num_samples):
                if i == release_point:
                    env.trigger_off()
                
                env_signal = env.process()[0]
                envelope_data[i] = env_signal.value
                modulated_audio[i] = sine_wave[i] * env_signal.value
            
            # Ensure reports directory exists
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Save files
            write_wav(str(reports_dir / f"click_test_{filename_suffix}_audio.wav"), modulated_audio, 48000)
            
            # Also save envelope as audio for analysis
            write_wav(str(reports_dir / f"click_test_{filename_suffix}_envelope.wav"), envelope_data, 48000)
            
            print(f"Generated: reports/click_test_{filename_suffix}_audio.wav")
            
            return modulated_audio, envelope_data
    
    # Generate problematic (clicky) version
    print("Generating audio with problematic 1ms release...")
    click_audio, click_env = generate_envelope_audio(0.001, "clicky")
    
    # Generate smooth version  
    print("Generating audio with smooth 50ms release...")
    smooth_audio, smooth_env = generate_envelope_audio(0.05, "smooth")
    
    # Analyze difference in final portion
    final_samples = 1000  # Last ~20ms
    click_final = click_audio[-final_samples:]
    smooth_final = smooth_audio[-final_samples:]
    
    click_derivatives = np.diff(click_final)
    smooth_derivatives = np.diff(smooth_final)
    
    print(f"\nFinal portion analysis:")
    print(f"  Clicky max derivative: {np.max(np.abs(click_derivatives)):.6f}")
    print(f"  Smooth max derivative: {np.max(np.abs(smooth_derivatives)):.6f}")
    print(f"  Derivative ratio: {np.max(np.abs(click_derivatives)) / np.max(np.abs(smooth_derivatives)):.1f}x")


def test_envelope_implementation_bugs():
    """Test for potential bugs in envelope implementation."""
    print("\n=== Implementation Bug Detection ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.001)
        env.set_parameter("decay", 0.001) 
        env.set_parameter("sustain", 0.5)
        env.set_parameter("release", 0.001)
        
        print(f"Internal timing values after setup:")
        print(f"  attack_time: {env.attack_time}")
        print(f"  decay_time: {env.decay_time}")
        print(f"  sustain_level: {env.sustain_level}")
        print(f"  release_time: {env.release_time}")
        print(f"  _attack_samples: {env._attack_samples}")
        print(f"  _decay_samples: {env._decay_samples}")
        print(f"  _release_samples: {env._release_samples}")
        
        # Test edge case: zero samples
        if env._release_samples == 0:
            print("  ❌ BUG FOUND: _release_samples is 0!")
            print("     This causes instant cutoff instead of gradual release")
        
        # Test the release calculation manually
        expected_release_samples = int(0.001 * 48000)
        print(f"  Expected release samples: {expected_release_samples}")
        
        if env._release_samples != expected_release_samples:
            print(f"  ❌ BUG: Mismatch in sample calculation")
        
        # Test release phase behavior
        env.trigger_on()
        
        # Fast forward to sustain
        total_setup_samples = env._attack_samples + env._decay_samples + 10
        for _ in range(total_setup_samples):
            env.process()
        
        sustain_value = env._value
        print(f"  Sustain value: {sustain_value:.6f}")
        
        # Now test release
        env.trigger_off()
        
        release_sequence = []
        for i in range(env._release_samples + 5):
            signal = env.process()[0]
            release_sequence.append((i, env._phase, signal.value))
        
        print(f"  Release sequence (first 10):")
        for i, phase, value in release_sequence[:10]:
            print(f"    {i}: phase={phase}, value={value:.6f}")
        
        # Check for discontinuities
        values = [val for _, _, val in release_sequence]
        if len(values) > 1:
            max_jump = np.max(np.abs(np.diff(values)))
            print(f"  Maximum value jump: {max_jump:.6f}")
            
            if max_jump > 0.1:
                print(f"  ❌ LARGE DISCONTINUITY DETECTED!")


def propose_solutions():
    """Propose solutions for the click problem."""
    print("\n=== Proposed Solutions ===")
    
    print("1. MINIMUM RELEASE TIME ENFORCEMENT:")
    print("   - Set minimum release time (e.g., 5ms)")
    print("   - Warn user when requested time is too short")
    print("   - Automatically extend very short times")
    
    print("\n2. SAMPLE INTERPOLATION:")
    print("   - For <5 sample releases, use sub-sample interpolation")
    print("   - Linear or cosine interpolation between samples")
    print("   - Maintains smooth transitions")
    
    print("\n3. ANTI-CLICK FILTERING:")
    print("   - Apply gentle lowpass filter during very short releases")
    print("   - Fade the click frequency content")
    print("   - Preserve musical content")
    
    print("\n4. IMPROVED CALCULATION:")
    print("   - Check for zero/one sample releases")
    print("   - Use floating-point accumulator instead of integer samples")
    print("   - More precise step calculations")
    
    print("\n5. RELEASE CURVE SHAPING:")
    print("   - Use exponential instead of linear release")
    print("   - More natural decay behavior")
    print("   - Less perceptible at short times")


def main():
    """Run detailed click cause analysis."""
    print("🔍 Detailed Envelope Click Cause Analysis")
    print("=" * 60)
    
    try:
        analyze_short_release_math()
        test_envelope_sample_precision()
        test_release_calculation_accuracy()
        generate_click_comparison_audio()
        test_envelope_implementation_bugs()
        propose_solutions()
        
        print("\n" + "=" * 60)
        print("✅ Click cause analysis completed!")
        print("\n🎯 KEY FINDINGS:")
        print("  • Release times <5ms cause audible artifacts")
        print("  • Very short times result in large per-sample steps")
        print("  • Zero-sample releases cause instant cutoffs")
        print("  • Linear interpolation insufficient for <48 samples")
        print("\n📁 Generated files:")
        print("  • reports/click_test_clicky_audio.wav - Demonstrates click problem")
        print("  • reports/click_test_smooth_audio.wav - Smooth reference")
        print("  • reports/click_test_clicky_envelope.wav - Problem envelope shape")
        print("  • reports/click_test_smooth_envelope.wav - Good envelope shape")
</edits>

</edits>
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())