#!/usr/bin/env python3
"""
Demo script showcasing context-aware DSP functions.

This script demonstrates how the new context-aware DSP utilities work,
allowing DSP functions to automatically use the current synthesis context's
sample rate while maintaining backward compatibility.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import (
    synthesis_context, SynthEngine,
    generate_silence, write_wav,
    generate_silence_explicit, write_wav_explicit,
    get_context_sample_rate, has_context,
    Oscillator, EnvelopeADSR, VCA
)


def demo_basic_context_dsp():
    """Demonstrate basic context-aware DSP functions."""
    print("=== Basic Context-Aware DSP Demo ===")
    
    # Check context availability
    print(f"Context available initially: {has_context()}")
    
    # Using context-aware DSP functions
    with synthesis_context(sample_rate=48000):
        print(f"Context available in context: {has_context()}")
        print(f"Context sample rate: {get_context_sample_rate()}Hz")
        
        # Generate silence using context sample rate
        silence = generate_silence(0.5)  # 0.5 seconds
        print(f"Generated silence: {silence.shape[0]} samples at {get_context_sample_rate()}Hz")
        
        # Create a simple tone
        osc = Oscillator()
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("waveform", "sine")
        
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.2)
        env.set_parameter("sustain", 0.8)
        env.set_parameter("release", 0.3)
        
        vca = VCA()
        
        # Generate 1 second of audio
        sample_rate = get_context_sample_rate()
        num_samples = sample_rate  # 1 second
        audio = np.zeros(num_samples, dtype=np.float32)
        
        env.trigger_on()
        
        for i in range(num_samples):
            if i > num_samples * 0.7:  # Release at 70%
                if env._note_on:
                    env.trigger_off()
            
            osc_signal = osc.process()[0]
            env_signal = env.process()[0]
            output = vca.process([osc_signal, env_signal])[0]
            audio[i] = output.value
        
        # Write using context-aware function (no sample_rate needed!)
        write_wav("demo_context_tone.wav", audio)
        print("✅ Wrote demo_context_tone.wav using context sample rate")


def demo_explicit_vs_context():
    """Compare explicit vs context-aware DSP functions."""
    print("\n=== Explicit vs Context-Aware Comparison ===")
    
    # Explicit approach (traditional)
    print("\n1. Explicit approach:")
    silence_explicit = generate_silence_explicit(0.25, 44100)
    print(f"   Generated {silence_explicit.shape[0]} samples at 44.1kHz (explicit)")
    
    # Simple sine wave for testing
    sample_rate_explicit = 44100
    duration = 0.25
    num_samples = int(sample_rate_explicit * duration)
    t = np.linspace(0, duration, num_samples, False)
    audio_explicit = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    write_wav_explicit("demo_explicit.wav", audio_explicit, 44100, 16)
    print("   ✅ Wrote demo_explicit.wav (44.1kHz, explicit)")
    
    # Context-aware approach
    print("\n2. Context-aware approach:")
    with synthesis_context(sample_rate=96000):
        silence_context = generate_silence(0.25)  # No sample rate needed!
        print(f"   Generated {silence_context.shape[0]} samples at {get_context_sample_rate()}Hz (context)")
        
        # Same sine wave at high quality
        sample_rate_context = get_context_sample_rate()
        duration = 0.25
        num_samples = int(sample_rate_context * duration)
        t = np.linspace(0, duration, num_samples, False)
        audio_context = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        write_wav("demo_context.wav", audio_context)  # No sample rate needed!
        print("   ✅ Wrote demo_context.wav (96kHz, context)")


def demo_different_contexts():
    """Demonstrate DSP functions in different contexts."""
    print("\n=== Different Context Sample Rates Demo ===")
    
    # Generate the same content at different sample rates
    sample_rates = [22050, 44100, 48000, 96000]
    
    for rate in sample_rates:
        print(f"\nGenerating content at {rate}Hz:")
        
        with synthesis_context(sample_rate=rate):
            # Generate 0.1 seconds of silence
            silence = generate_silence(0.1)
            
            # Generate 0.2 seconds of 1kHz tone
            duration = 0.2
            num_samples = int(get_context_sample_rate() * duration)
            t = np.linspace(0, duration, num_samples, False)
            tone = 0.3 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
            
            # Combine silence + tone
            combined = np.concatenate([silence, tone])
            
            # Write using context (no sample rate needed!)
            filename = f"demo_tone_{rate}Hz.wav"
            write_wav(filename, combined)
            
            print(f"   ✅ Generated {filename}: {combined.shape[0]} samples")


def demo_nested_contexts():
    """Demonstrate DSP functions in nested contexts."""
    print("\n=== Nested Contexts DSP Demo ===")
    
    with synthesis_context(sample_rate=48000):
        print(f"Outer context: {get_context_sample_rate()}Hz")
        
        # Generate some audio in outer context
        silence1 = generate_silence(0.1)
        print(f"   Outer silence: {silence1.shape[0]} samples")
        
        with synthesis_context(sample_rate=22050):
            print(f"  Inner context: {get_context_sample_rate()}Hz")
            
            # Generate audio in inner context
            silence2 = generate_silence(0.1)
            print(f"     Inner silence: {silence2.shape[0]} samples")
            
            # Write from inner context
            tone_inner = 0.2 * np.sin(2 * np.pi * 800 * np.linspace(0, 0.1, silence2.shape[0], False))
            write_wav("demo_nested_inner.wav", tone_inner.astype(np.float32))
            print("     ✅ Wrote demo_nested_inner.wav (22.05kHz)")
        
        # Back to outer context
        print(f"Back to outer: {get_context_sample_rate()}Hz")
        tone_outer = 0.2 * np.sin(2 * np.pi * 400 * np.linspace(0, 0.1, silence1.shape[0], False))
        write_wav("demo_nested_outer.wav", tone_outer.astype(np.float32))
        print("   ✅ Wrote demo_nested_outer.wav (48kHz)")


def demo_error_handling():
    """Demonstrate error handling for context-aware DSP."""
    print("\n=== Error Handling Demo ===")
    
    # Try using context-aware functions without context
    print("\n1. Using context-aware functions without context:")
    try:
        silence = generate_silence(0.1)  # Should fail
        print("   ❌ Unexpected success")
    except Exception as e:
        print(f"   ✅ Expected error: {type(e).__name__}: {e}")
    
    # Try writing without context
    print("\n2. Writing audio without context:")
    try:
        dummy_audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        write_wav("should_fail.wav", dummy_audio)  # Should fail
        print("   ❌ Unexpected success")
    except Exception as e:
        print(f"   ✅ Expected error: {type(e).__name__}: {e}")
    
    # Show explicit functions still work
    print("\n3. Explicit functions work without context:")
    try:
        silence_explicit = generate_silence_explicit(0.05, 48000)
        dummy_audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        write_wav_explicit("demo_explicit_works.wav", dummy_audio, 48000)
        print("   ✅ Explicit functions work fine without context")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


def demo_engine_context_dsp():
    """Demonstrate DSP functions with SynthEngine context."""
    print("\n=== SynthEngine Context DSP Demo ===")
    
    # Using SynthEngine as context manager
    with SynthEngine(sample_rate=44100) as engine:
        print(f"Engine context sample rate: {get_context_sample_rate()}Hz")
        
        # Create synthesizer modules
        osc = Oscillator()
        osc.set_parameter("frequency", 523.25)  # C5
        osc.set_parameter("waveform", "sawtooth")
        
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.02)
        env.set_parameter("decay", 0.1)
        env.set_parameter("sustain", 0.6)
        env.set_parameter("release", 0.4)
        
        vca = VCA()
        
        # Generate 0.8 seconds of audio
        duration = 0.8
        sample_rate = get_context_sample_rate()
        num_samples = int(sample_rate * duration)
        audio = np.zeros(num_samples, dtype=np.float32)
        
        env.trigger_on()
        
        for i in range(num_samples):
            if i > num_samples * 0.6:  # Release at 60%
                if env._note_on:
                    env.trigger_off()
            
            osc_signal = osc.process()[0]
            env_signal = env.process()[0]
            output = vca.process([osc_signal, env_signal])[0]
            audio[i] = output.value
        
        # Add some silence at the beginning and end
        lead_silence = generate_silence(0.1)
        tail_silence = generate_silence(0.1)
        
        # Combine all parts
        full_audio = np.concatenate([lead_silence, audio, tail_silence])
        
        # Write using context-aware function
        write_wav("demo_engine_context.wav", full_audio)
        print("✅ Generated demo_engine_context.wav using SynthEngine context")


def demo_practical_usage():
    """Demonstrate practical usage patterns."""
    print("\n=== Practical Usage Patterns Demo ===")
    
    def create_drum_hit(frequency: float, duration: float):
        """Factory function that uses current context."""
        # This function assumes it's called within a synthesis context
        sample_rate = get_context_sample_rate()
        num_samples = int(sample_rate * duration)
        
        # Create noise-based drum sound
        osc = Oscillator()
        osc.set_parameter("frequency", frequency)
        osc.set_parameter("waveform", "noise")
        
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.001)
        env.set_parameter("decay", duration * 0.8)
        env.set_parameter("sustain", 0.0)
        env.set_parameter("release", duration * 0.2)
        
        vca = VCA()
        
        audio = np.zeros(num_samples, dtype=np.float32)
        env.trigger_on()
        
        for i in range(num_samples):
            osc_signal = osc.process()[0]
            env_signal = env.process()[0]
            output = vca.process([osc_signal, env_signal])[0]
            audio[i] = output.value * 0.8  # Scale down
        
        return audio
    
    def create_pattern():
        """Create a simple drum pattern."""
        # Kick drum (low frequency)
        kick = create_drum_hit(60.0, 0.3)
        
        # Snare drum (higher frequency)
        snare = create_drum_hit(200.0, 0.2)
        
        # Hi-hat (very high frequency, short)
        hihat = create_drum_hit(8000.0, 0.05)
        
        # Create pattern: kick, hihat, snare, hihat
        gap = generate_silence(0.05)  # Small gap between sounds
        pattern = np.concatenate([kick, gap, hihat, gap, snare, gap, hihat])
        
        return pattern
    
    # Create patterns at different sample rates
    print("\nCreating drum patterns at different quality levels:")
    
    # Lo-fi pattern
    with synthesis_context(sample_rate=22050):
        lofi_pattern = create_pattern()
        write_wav("demo_drums_lofi.wav", lofi_pattern)
        print(f"   ✅ Lo-fi pattern: {lofi_pattern.shape[0]} samples at 22.05kHz")
    
    # Standard pattern
    with synthesis_context(sample_rate=48000):
        standard_pattern = create_pattern()
        write_wav("demo_drums_standard.wav", standard_pattern)
        print(f"   ✅ Standard pattern: {standard_pattern.shape[0]} samples at 48kHz")
    
    # High-quality pattern
    with synthesis_context(sample_rate=96000):
        hq_pattern = create_pattern()
        write_wav("demo_drums_hq.wav", hq_pattern)
        print(f"   ✅ High-quality pattern: {hq_pattern.shape[0]} samples at 96kHz")


def main():
    """Run all context-aware DSP demonstrations."""
    print("🎵 Signals Context-Aware DSP Functions Demo")
    print("=" * 60)
    
    try:
        demo_basic_context_dsp()
        demo_explicit_vs_context()
        demo_different_contexts()
        demo_nested_contexts()
        demo_error_handling()
        demo_engine_context_dsp()
        demo_practical_usage()
        
        print("\n" + "=" * 60)
        print("✅ All context-aware DSP demonstrations completed successfully!")
        print("\nKey Benefits of Context-Aware DSP Functions:")
        print("  • 🎯 Automatic sample rate from context (no manual passing)")
        print("  • 🔒 Guaranteed consistency (all DSP uses same sample rate)")
        print("  • 🔄 Easy quality switching (change context, all DSP follows)")
        print("  • 🔙 Full backward compatibility (explicit functions available)")
        print("  • 🏗️  Clean factory functions (no sample rate management)")
        print("  • ⚡ Zero performance overhead")
        print("  • 🛡️  Clear error messages when context missing")
        
        print("\nFiles generated:")
        generated_files = [
            "demo_context_tone.wav - Context-aware tone generation",
            "demo_explicit.wav - Traditional explicit approach (44.1kHz)",
            "demo_context.wav - Context-aware approach (96kHz)",
            "demo_tone_22050Hz.wav - Low quality tone",
            "demo_tone_44100Hz.wav - CD quality tone", 
            "demo_tone_48000Hz.wav - Professional quality tone",
            "demo_tone_96000Hz.wav - High resolution tone",
            "demo_nested_inner.wav - Inner context audio (22.05kHz)",
            "demo_nested_outer.wav - Outer context audio (48kHz)",
            "demo_explicit_works.wav - Explicit function demo",
            "demo_engine_context.wav - SynthEngine context demo",
            "demo_drums_lofi.wav - Lo-fi drum pattern (22.05kHz)",
            "demo_drums_standard.wav - Standard drum pattern (48kHz)",
            "demo_drums_hq.wav - High-quality drum pattern (96kHz)"
        ]
        
        for file_desc in generated_files:
            print(f"  • {file_desc}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())