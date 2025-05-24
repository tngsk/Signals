#!/usr/bin/env python3
"""
Demo script showcasing context-based sample rate management.

This script demonstrates the new SynthContext system that allows modules
to be created without explicit sample rate specification while maintaining
consistency across the synthesis graph.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import (
    SynthEngine, SynthContext, synthesis_context,
    Oscillator, EnvelopeADSR, VCA, Mixer,
    write_wav
)
import numpy as np


def demo_basic_context_usage():
    """Demonstrate basic context usage patterns."""
    print("=== Basic Context Usage Demo ===")
    
    # Old way: explicit sample rates everywhere
    print("\n1. Old way (still supported):")
    sample_rate = 48000
    osc_old = Oscillator(sample_rate=sample_rate)
    env_old = EnvelopeADSR(sample_rate=sample_rate)
    vca_old = VCA(sample_rate=sample_rate)
    print(f"   Oscillator: {osc_old.sample_rate}Hz")
    print(f"   Envelope: {env_old.sample_rate}Hz") 
    print(f"   VCA: {vca_old.sample_rate}Hz")
    
    # New way: context-based
    print("\n2. New way (context-based):")
    with SynthContext(sample_rate=48000):
        osc_new = Oscillator()  # No sample_rate needed!
        env_new = EnvelopeADSR()  # No sample_rate needed!
        vca_new = VCA()  # No sample_rate needed!
        print(f"   Oscillator: {osc_new.sample_rate}Hz")
        print(f"   Envelope: {env_new.sample_rate}Hz")
        print(f"   VCA: {vca_new.sample_rate}Hz")
    
    print("✅ All modules automatically use context sample rate!")


def demo_engine_context():
    """Demonstrate SynthEngine as context manager."""
    print("\n=== SynthEngine Context Demo ===")
    
    # SynthEngine as context manager
    print("\n1. Using SynthEngine as context manager:")
    with SynthEngine(sample_rate=44100) as engine:
        # Modules created here automatically use engine's sample rate
        osc = Oscillator()
        env = EnvelopeADSR()
        mixer = Mixer()
        
        print(f"   Engine sample rate: {engine.sample_rate}Hz")
        print(f"   Oscillator: {osc.sample_rate}Hz")
        print(f"   Envelope: {env.sample_rate}Hz")
        print(f"   Mixer: doesn't use sample rate (OK)")
    
    # Using engine.context() method
    print("\n2. Using engine.context() method:")
    engine = SynthEngine(sample_rate=96000)
    with engine.context():
        osc_hq = Oscillator()
        env_hq = EnvelopeADSR()
        
        print(f"   High-quality Oscillator: {osc_hq.sample_rate}Hz")
        print(f"   High-quality Envelope: {env_hq.sample_rate}Hz")


def demo_different_sample_rates():
    """Demonstrate easy switching between sample rates."""
    print("\n=== Different Sample Rates Demo ===")
    
    sample_rates = [22050, 44100, 48000, 96000]
    modules = {}
    
    for rate in sample_rates:
        print(f"\nCreating modules at {rate}Hz:")
        with synthesis_context(sample_rate=rate):
            osc = Oscillator()
            env = EnvelopeADSR()
            modules[rate] = (osc, env)
            print(f"   ✅ Oscillator and Envelope created at {rate}Hz")
    
    # Verify all modules have correct sample rates
    print("\nVerification:")
    for rate, (osc, env) in modules.items():
        print(f"   {rate}Hz: osc={osc.sample_rate}Hz, env={env.sample_rate}Hz")


def demo_nested_contexts():
    """Demonstrate nested context behavior."""
    print("\n=== Nested Contexts Demo ===")
    
    with SynthContext(sample_rate=48000):
        osc1 = Oscillator()
        print(f"Outer context - Oscillator: {osc1.sample_rate}Hz")
        
        with SynthContext(sample_rate=44100):
            osc2 = Oscillator()
            print(f"  Inner context - Oscillator: {osc2.sample_rate}Hz")
            
            with SynthContext(sample_rate=96000):
                osc3 = Oscillator()
                print(f"    Deepest context - Oscillator: {osc3.sample_rate}Hz")
            
            print(f"  Back to inner - New oscillator: {Oscillator().sample_rate}Hz")
        
        print(f"Back to outer - New oscillator: {Oscillator().sample_rate}Hz")


def demo_explicit_override():
    """Demonstrate explicit sample rate override."""
    print("\n=== Explicit Override Demo ===")
    
    with SynthContext(sample_rate=48000):
        osc_context = Oscillator()  # Uses context
        osc_explicit = Oscillator(sample_rate=22050)  # Explicit override
        
        print(f"Context oscillator: {osc_context.sample_rate}Hz")
        print(f"Explicit oscillator: {osc_explicit.sample_rate}Hz")
        print("✅ Explicit sample rate overrides context")


def demo_module_factory_pattern():
    """Demonstrate module factory pattern with context."""
    print("\n=== Module Factory Pattern Demo ===")
    
    def create_basic_synth():
        """Factory function that creates a basic synth setup."""
        # This function assumes it's called within a synthesis context
        osc = Oscillator()
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("waveform", "sine")
        
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.2)
        env.set_parameter("sustain", 0.7)
        env.set_parameter("release", 0.3)
        
        vca = VCA()
        
        return osc, env, vca
    
    def create_drum_synth():
        """Factory for drum synthesis."""
        osc = Oscillator()
        osc.set_parameter("frequency", 60.0)  # Low frequency
        osc.set_parameter("waveform", "noise")
        
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.001)  # Very fast attack
        env.set_parameter("decay", 0.1)
        env.set_parameter("sustain", 0.0)   # No sustain
        env.set_parameter("release", 0.05)  # Quick release
        
        vca = VCA()
        
        return osc, env, vca
    
    print("\nCreating different synths at different sample rates:")
    
    # Melodic synth at high quality
    with synthesis_context(96000):
        osc_m, env_m, vca_m = create_basic_synth()
        print(f"   Melodic synth: {osc_m.sample_rate}Hz (high quality)")
    
    # Drum synth at standard quality
    with synthesis_context(48000):
        osc_d, env_d, vca_d = create_drum_synth()
        print(f"   Drum synth: {osc_d.sample_rate}Hz (standard quality)")
    
    # Demo synth at low quality
    with synthesis_context(22050):
        osc_demo, env_demo, vca_demo = create_basic_synth()
        print(f"   Demo synth: {osc_demo.sample_rate}Hz (low quality)")


def demo_audio_generation():
    """Demonstrate audio generation with context system."""
    print("\n=== Audio Generation Demo ===")
    
    def generate_tone(frequency, duration, sample_rate):
        """Generate a simple tone using context system."""
        with synthesis_context(sample_rate):
            # Create modules without specifying sample rate
            osc = Oscillator()
            osc.set_parameter("frequency", frequency)
            osc.set_parameter("waveform", "sine")
            
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.05)
            env.set_parameter("decay", 0.1)
            env.set_parameter("sustain", 0.8)
            env.set_parameter("release", 0.2)
            
            vca = VCA()
            
            # Generate audio
            num_samples = int(sample_rate * duration)
            audio = np.zeros(num_samples)
            
            env.trigger_on()
            
            for i in range(num_samples):
                # Release envelope partway through
                if i > num_samples * 0.7:
                    if env._note_on:  # Only trigger off once
                        env.trigger_off()
                
                osc_signal = osc.process()[0]
                env_signal = env.process()[0]
                output = vca.process([osc_signal, env_signal])[0]
                audio[i] = output.value
        
        return audio
    
    # Generate the same 440Hz tone at different sample rates
    print("\nGenerating 440Hz tone at different sample rates:")
    
    # Low quality
    audio_22k = generate_tone(440.0, 0.5, 22050)
    write_wav("demo_tone_22k.wav", audio_22k, 22050)
    print("   ✅ Generated demo_tone_22k.wav (22.05kHz)")
    
    # Standard quality
    audio_48k = generate_tone(440.0, 0.5, 48000)
    write_wav("demo_tone_48k.wav", audio_48k, 48000)
    print("   ✅ Generated demo_tone_48k.wav (48kHz)")
    
    # High quality
    audio_96k = generate_tone(440.0, 0.5, 96000)
    write_wav("demo_tone_96k.wav", audio_96k, 96000)
    print("   ✅ Generated demo_tone_96k.wav (96kHz)")


def demo_error_handling():
    """Demonstrate error handling in context system."""
    print("\n=== Error Handling Demo ===")
    
    # Try to create module without context
    print("\n1. Creating module without context (uses default):")
    try:
        osc = Oscillator()  # Should use default sample rate
        print(f"   ✅ Success: Oscillator created with {osc.sample_rate}Hz (default)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Try to access context when none exists
    print("\n2. Accessing context when none exists:")
    try:
        from signals import SynthContext
        rate = SynthContext.get_sample_rate()  # Should raise error
        print(f"   Unexpected: Got sample rate {rate}")
    except Exception as e:
        print(f"   ✅ Expected error: {e}")
    
    # Context cleanup after exception
    print("\n3. Context cleanup after exception:")
    try:
        with SynthContext(sample_rate=48000):
            print("   Inside context...")
            raise ValueError("Test exception")
    except ValueError:
        print("   ✅ Exception caught")
    
    # Verify context was cleaned up
    try:
        from signals import SynthContext
        SynthContext.get_sample_rate()
        print("   ❌ Context not cleaned up!")
    except:
        print("   ✅ Context properly cleaned up after exception")


def main():
    """Run all context usage demonstrations."""
    print("🎵 Signals Context-Based Sample Rate Management Demo")
    print("=" * 60)
    
    try:
        demo_basic_context_usage()
        demo_engine_context()
        demo_different_sample_rates()
        demo_nested_contexts()
        demo_explicit_override()
        demo_module_factory_pattern()
        demo_audio_generation()
        demo_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All context management demonstrations completed successfully!")
        print("\nKey Benefits of Context-Based Sample Rate Management:")
        print("  • 🎯 Simplified module creation (no sample_rate parameter needed)")
        print("  • 🔒 Guaranteed consistency (all modules use same sample rate)")
        print("  • 🔄 Easy sample rate switching for different quality levels")
        print("  • 🏗️  Clean factory patterns for module creation")
        print("  • 🔙 Full backward compatibility with explicit sample rates")
        print("  • 🧵 Thread-safe context isolation")
        print("  • ⚡ High performance with minimal overhead")
        print("\nFiles generated:")
        print("  • demo_tone_22k.wav - 22.05kHz quality tone")
        print("  • demo_tone_48k.wav - 48kHz quality tone") 
        print("  • demo_tone_96k.wav - 96kHz quality tone")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())