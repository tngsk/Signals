#!/usr/bin/env python3
"""
Envelope click detection and analysis script.

This script performs comprehensive testing of the EnvelopeADSR module
to identify and analyze click artifacts that occur at the end of audio.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import EnvelopeADSR, SynthContext, write_wav
import time


def test_envelope_basic_behavior():
    """Test basic envelope behavior and timing."""
    print("=== Basic Envelope Behavior Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.2)
        env.set_parameter("sustain", 0.7)
        env.set_parameter("release", 0.3)
        
        # Test envelope phases
        print(f"Initial state: phase={env._phase}, value={env._value}")
        
        # Trigger on
        env.trigger_on()
        print(f"After trigger_on: phase={env._phase}, value={env._value}")
        
        # Process attack phase
        attack_samples = int(0.1 * 48000)  # 0.1s at 48kHz
        print(f"Expected attack samples: {attack_samples}")
        
        values = []
        for i in range(attack_samples + 10):  # Process a bit extra
            signal = env.process()[0]
            values.append(signal.value)
            if i < 5 or i > attack_samples - 5:
                print(f"Sample {i}: phase={env._phase}, value={signal.value:.6f}")
        
        print(f"Attack phase completed at sample {len(values)}")
        print(f"Final attack value: {values[attack_samples-1]:.6f}")
        
        return values


def test_envelope_transitions():
    """Test envelope phase transitions for smoothness."""
    print("\n=== Envelope Transition Smoothness Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.05)   # 50ms
        env.set_parameter("decay", 0.1)     # 100ms
        env.set_parameter("sustain", 0.6)   # 60%
        env.set_parameter("release", 0.2)   # 200ms
        
        env.trigger_on()
        
        values = []
        phases = []
        times = []
        
        # Process full envelope cycle
        total_samples = int(1.0 * 48000)  # 1 second
        release_triggered = False
        
        for i in range(total_samples):
            # Trigger release after 500ms
            if i == int(0.5 * 48000) and not release_triggered:
                env.trigger_off()
                release_triggered = True
                print(f"Triggered release at sample {i}, current value: {env._value:.6f}")
            
            signal = env.process()[0]
            values.append(signal.value)
            phases.append(env._phase)
            times.append(i / 48000)
        
        # Analyze transitions
        print(f"Envelope processing complete. Total samples: {len(values)}")
        print(f"Final value: {values[-1]:.6f}")
        print(f"Final phase: {phases[-1]}")
        
        # Check for abrupt changes (potential clicks)
        derivatives = np.diff(values)
        max_derivative = np.max(np.abs(derivatives))
        large_jumps = np.where(np.abs(derivatives) > 0.01)[0]
        
        print(f"Maximum derivative: {max_derivative:.6f}")
        print(f"Large jumps (>0.01): {len(large_jumps)} locations")
        
        if len(large_jumps) > 0:
            print("Large jump locations:")
            for jump_idx in large_jumps[:10]:  # Show first 10
                print(f"  Sample {jump_idx}: {values[jump_idx]:.6f} -> {values[jump_idx+1]:.6f} (diff: {derivatives[jump_idx]:.6f})")
        
        return values, phases, times, derivatives


def test_envelope_end_behavior():
    """Test envelope behavior at the end of processing."""
    print("\n=== Envelope End Behavior Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.02)
        env.set_parameter("decay", 0.05)
        env.set_parameter("sustain", 0.5)
        env.set_parameter("release", 0.1)
        
        env.trigger_on()
        
        # Process until sustain
        sustain_start = int((0.02 + 0.05) * 48000)  # After attack + decay
        for i in range(sustain_start + 100):
            env.process()
        
        print(f"Sustain phase value: {env._value:.6f}")
        
        # Trigger release
        env.trigger_off()
        print(f"Release triggered, phase: {env._phase}")
        
        # Process release phase carefully
        release_samples = int(0.1 * 48000)  # 100ms release
        end_values = []
        
        for i in range(release_samples + 50):  # Process a bit extra
            signal = env.process()[0]
            end_values.append(signal.value)
            
            # Log final samples
            if i > release_samples - 10:
                print(f"Release sample {i}: phase={env._phase}, value={signal.value:.6f}")
        
        # Check if envelope properly reaches zero
        final_values = end_values[-10:]
        print(f"Final 10 values: {[f'{v:.6f}' for v in final_values]}")
        
        # Check for non-zero values after release should be complete
        non_zero_after_release = [v for v in end_values[release_samples:] if abs(v) > 1e-6]
        if non_zero_after_release:
            print(f"WARNING: Non-zero values after release completion: {non_zero_after_release}")
        else:
            print("✅ Envelope properly reaches zero after release")
        
        return end_values


def test_envelope_audio_generation():
    """Generate audio with envelope to detect clicks."""
    print("\n=== Audio Generation Click Test ===")
    
    with SynthContext(sample_rate=48000):
        # Create a simple oscillator manually for testing
        def generate_sine(freq, sample_rate, num_samples):
            t = np.arange(num_samples) / sample_rate
            return 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Generate base tone
        duration = 1.0
        num_samples = int(duration * 48000)
        base_audio = generate_sine(440.0, 48000, num_samples)
        
        # Apply envelope
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.05)
        env.set_parameter("decay", 0.1)
        env.set_parameter("sustain", 0.7)
        env.set_parameter("release", 0.2)
        
        env.trigger_on()
        
        modulated_audio = np.zeros(num_samples)
        envelope_values = np.zeros(num_samples)
        
        # Trigger release at 70% through
        release_point = int(0.7 * num_samples)
        
        for i in range(num_samples):
            if i == release_point:
                env.trigger_off()
            
            env_signal = env.process()[0]
            envelope_values[i] = env_signal.value
            modulated_audio[i] = base_audio[i] * env_signal.value
        
        # Analyze for clicks
        print(f"Generated {num_samples} samples")
        print(f"Final envelope value: {envelope_values[-1]:.6f}")
        print(f"Final audio value: {modulated_audio[-1]:.6f}")
        
        # Check for abrupt changes in final portion
        final_portion = modulated_audio[-1000:]  # Last 1000 samples (~20ms)
        final_derivatives = np.diff(final_portion)
        max_final_derivative = np.max(np.abs(final_derivatives))
        
        print(f"Maximum derivative in final 20ms: {max_final_derivative:.6f}")
        
        # Save audio files for analysis
        write_wav("envelope_test_modulated.wav", modulated_audio, 48000)
        write_wav("envelope_test_base.wav", base_audio, 48000)
        
        print("✅ Audio files generated: envelope_test_modulated.wav, envelope_test_base.wav")
        
        return modulated_audio, envelope_values, base_audio


def test_envelope_zero_crossing():
    """Test envelope behavior around zero crossings."""
    print("\n=== Zero Crossing Test ===")
    
    with SynthContext(sample_rate=48000):
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.01)
        env.set_parameter("decay", 0.02)
        env.set_parameter("sustain", 0.001)  # Very low sustain
        env.set_parameter("release", 0.05)
        
        env.trigger_on()
        
        values = []
        phases = []
        
        # Process until well past release
        total_samples = int(0.2 * 48000)  # 200ms
        release_triggered = False
        
        for i in range(total_samples):
            if i == int(0.05 * 48000) and not release_triggered:
                env.trigger_off()
                release_triggered = True
            
            signal = env.process()[0]
            values.append(signal.value)
            phases.append(env._phase)
        
        # Find where envelope reaches zero
        zero_crossings = []
        for i in range(1, len(values)):
            if values[i-1] > 0 and values[i] == 0:
                zero_crossings.append(i)
        
        print(f"Zero crossings found at samples: {zero_crossings}")
        
        # Check behavior after zero crossing
        if zero_crossings:
            first_zero = zero_crossings[0]
            post_zero_values = values[first_zero:first_zero+10]
            post_zero_phases = phases[first_zero:first_zero+10]
            
            print(f"Values after first zero crossing:")
            for i, (val, phase) in enumerate(zip(post_zero_values, post_zero_phases)):
                print(f"  Sample {first_zero + i}: value={val:.6f}, phase={phase}")
            
            # Check if envelope stays at zero
            non_zero_after_zero = [v for v in post_zero_values if v != 0]
            if non_zero_after_zero:
                print(f"WARNING: Non-zero values after zero crossing: {non_zero_after_zero}")
            else:
                print("✅ Envelope stays at zero after crossing")
        
        return values, phases


def test_different_release_times():
    """Test different release times for click analysis."""
    print("\n=== Different Release Times Test ===")
    
    release_times = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    for release_time in release_times:
        print(f"\nTesting release time: {release_time}s")
        
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.02)
            env.set_parameter("decay", 0.05)
            env.set_parameter("sustain", 0.8)
            env.set_parameter("release", release_time)
            
            env.trigger_on()
            
            # Process to sustain
            sustain_samples = int(0.1 * 48000)
            for _ in range(sustain_samples):
                env.process()
            
            # Trigger release and process
            env.trigger_off()
            release_samples = int(release_time * 48000) + 100
            
            release_values = []
            for i in range(release_samples):
                signal = env.process()[0]
                release_values.append(signal.value)
            
            # Analyze release curve smoothness
            if len(release_values) > 1:
                derivatives = np.diff(release_values)
                max_derivative = np.max(np.abs(derivatives))
                
                # Check final values
                final_value = release_values[-1]
                
                print(f"  Max derivative: {max_derivative:.6f}")
                print(f"  Final value: {final_value:.6f}")
                print(f"  Values near end: {[f'{v:.6f}' for v in release_values[-5:]]}")
                
                # Look for abrupt changes
                large_jumps = np.where(np.abs(derivatives) > 0.01)[0]
                if len(large_jumps) > 0:
                    print(f"  WARNING: {len(large_jumps)} large jumps detected")


def analyze_envelope_data(values, phases, times, derivatives):
    """Analyze envelope data numerically without plotting."""
    print("\n=== Numerical Analysis ===")
    
    # Basic statistics
    print(f"Total samples: {len(values)}")
    print(f"Duration: {times[-1]:.3f}s")
    print(f"Min value: {np.min(values):.6f}")
    print(f"Max value: {np.max(values):.6f}")
    print(f"Final value: {values[-1]:.6f}")
    
    # Derivative analysis
    max_derivative = np.max(np.abs(derivatives))
    mean_derivative = np.mean(np.abs(derivatives))
    print(f"Max derivative: {max_derivative:.6f}")
    print(f"Mean derivative: {mean_derivative:.6f}")
    
    # Find large jumps
    large_jumps = np.where(np.abs(derivatives) > 0.01)[0]
    print(f"Large jumps (>0.01): {len(large_jumps)}")
    
    # Phase analysis
    unique_phases = np.unique(phases)
    print(f"Phases encountered: {unique_phases}")
    
    # Final portion analysis
    final_samples = min(1000, len(values) // 4)
    final_values = values[-final_samples:]
    final_range = np.max(final_values) - np.min(final_values)
    print(f"Final {final_samples} samples range: {final_range:.6f}")
    
    return {
        'max_derivative': max_derivative,
        'mean_derivative': mean_derivative,
        'large_jumps': len(large_jumps),
        'final_range': final_range,
        'final_value': values[-1]
    }


def main():
    """Run all envelope click detection tests."""
    print("🔍 Envelope Click Detection and Analysis")
    print("=" * 60)
    
    try:
        # Basic behavior test
        basic_values = test_envelope_basic_behavior()
        
        # Transition smoothness test
        values, phases, times, derivatives = test_envelope_transitions()
        
        # End behavior test
        end_values = test_envelope_end_behavior()
        
        # Audio generation test
        audio, env_values, base_audio = test_envelope_audio_generation()
        
        # Zero crossing test
        zero_values, zero_phases = test_envelope_zero_crossing()
        
        # Different release times test
        test_different_release_times()
        
        # Generate numerical analysis
        analysis_results = analyze_envelope_data(values, phases, times, derivatives)
        
        print("\n" + "=" * 60)
        print("✅ Envelope click analysis completed!")
        print("\nGenerated files:")
        print("  • envelope_test_modulated.wav - Audio with envelope applied")
        print("  • envelope_test_base.wav - Base sine wave")
        print("\nLook for:")
        print("  • Large derivatives indicating abrupt changes")
        print("  • Non-zero values after envelope should be silent")
        print("  • Discontinuities in the envelope shape")
        print("  • Click artifacts in the audio files")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())