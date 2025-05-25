#!/usr/bin/env python3
"""
Filter Module Demonstration

This example demonstrates the Filter module capabilities including:
- Different filter types (lowpass, highpass, bandpass, notch)
- Real-time parameter changes
- Filter frequency sweeps
- Integration with oscillators and envelopes

The example generates several audio files showcasing different filtering effects.
"""

import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signals import (
    SynthContext, Oscillator, Filter, EnvelopeADSR, VCA, OutputWav,
    WaveformType, FilterType, Signal, write_wav
)


def create_output_dir():
    """Create output directory for audio files."""
    output_dir = "filter_examples"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def demo_filter_types():
    """Demonstrate different filter types with the same input signal."""
    print("Generating filter type comparison...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        # Create a complex harmonic input signal
        osc = Oscillator(waveform=WaveformType.SAW)
        osc.set_parameter("frequency", 220.0)  # A3
        osc.set_parameter("amplitude", 0.8)
        
        filter_types = [
            (FilterType.LOWPASS, "lowpass"),
            (FilterType.HIGHPASS, "highpass"), 
            (FilterType.BANDPASS, "bandpass"),
            (FilterType.NOTCH, "notch")
        ]
        
        for filter_type, name in filter_types:
            # Create filter with moderate settings
            filt = Filter(filter_type=filter_type)
            filt.set_parameter("cutoff_frequency", 880.0)  # A5 - 2 octaves up
            filt.set_parameter("resonance", 2.0)
            
            # Generate audio
            audio_buffer = []
            duration_samples = int(2.0 * 48000)  # 2 seconds
            
            for _ in range(duration_samples):
                osc_signal = osc.process()[0]
                filtered_signal = filt.process([osc_signal])[0]
                audio_buffer.append(filtered_signal.value)
            
            # Write to file
            filename = f"{output_dir}/filter_{name}_demo.wav"
            write_wav(filename, np.array(audio_buffer), sample_rate=48000)
            print(f"  Generated: {filename}")


def demo_frequency_sweep():
    """Demonstrate filter cutoff frequency sweep."""
    print("Generating frequency sweep demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        # Rich harmonic content for filtering
        osc = Oscillator(waveform=WaveformType.SAW)
        osc.set_parameter("frequency", 110.0)  # A2
        osc.set_parameter("amplitude", 0.7)
        
        # Lowpass filter with high resonance
        filt = Filter(filter_type=FilterType.LOWPASS)
        filt.set_parameter("resonance", 5.0)  # High Q for dramatic effect
        
        audio_buffer = []
        duration = 8.0  # 8 seconds
        duration_samples = int(duration * 48000)
        
        for i in range(duration_samples):
            # Logarithmic frequency sweep from 100Hz to 8kHz
            progress = i / duration_samples
            cutoff = 100.0 * (8000.0 / 100.0) ** progress
            filt.set_parameter("cutoff_frequency", cutoff)
            
            osc_signal = osc.process()[0]
            filtered_signal = filt.process([osc_signal])[0]
            audio_buffer.append(filtered_signal.value)
        
        filename = f"{output_dir}/filter_frequency_sweep.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_resonance_sweep():
    """Demonstrate filter resonance sweep."""
    print("Generating resonance sweep demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SAW)
        osc.set_parameter("frequency", 220.0)
        osc.set_parameter("amplitude", 0.6)
        
        filt = Filter(filter_type=FilterType.LOWPASS)
        filt.set_parameter("cutoff_frequency", 1000.0)  # Fixed cutoff
        
        audio_buffer = []
        duration = 6.0
        duration_samples = int(duration * 48000)
        
        for i in range(duration_samples):
            # Resonance sweep from 0.5 to 8.0
            progress = i / duration_samples
            resonance = 0.5 + progress * 7.5
            filt.set_parameter("resonance", resonance)
            
            osc_signal = osc.process()[0]
            filtered_signal = filt.process([osc_signal])[0]
            audio_buffer.append(filtered_signal.value)
        
        filename = f"{output_dir}/filter_resonance_sweep.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_envelope_controlled_filter():
    """Demonstrate envelope-controlled filter cutoff."""
    print("Generating envelope-controlled filter demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SAW)
        osc.set_parameter("frequency", 330.0)  # E4
        osc.set_parameter("amplitude", 0.8)
        
        env = EnvelopeADSR()
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.3)
        env.set_parameter("sustain", 0.4)
        env.set_parameter("release", 0.8)
        
        filt = Filter(filter_type=FilterType.LOWPASS)
        filt.set_parameter("resonance", 3.0)
        
        vca = VCA()
        
        audio_buffer = []
        duration = 3.0
        duration_samples = int(duration * 48000)
        
        # Trigger envelope at start
        env.trigger_on()
        
        # Release after 1.5 seconds
        release_sample = int(1.5 * 48000)
        
        for i in range(duration_samples):
            if i == release_sample:
                env.trigger_off()
            
            # Get envelope value for filter modulation
            env_signal = env.process()[0]
            env_value = env_signal.value
            
            # Use envelope to control filter cutoff (200Hz to 4000Hz range)
            cutoff = 200.0 + env_value * 3800.0
            filt.set_parameter("cutoff_frequency", cutoff)
            
            # Process audio chain
            osc_signal = osc.process()[0]
            filtered_signal = filt.process([osc_signal])[0]
            final_signal = vca.process([filtered_signal, env_signal])[0]
            audio_buffer.append(final_signal.value)
        
        filename = f"{output_dir}/filter_envelope_controlled.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_filter_oscillation():
    """Demonstrate filter self-oscillation at high resonance."""
    print("Generating filter self-oscillation demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        # Very quiet input to hear self-oscillation
        osc = Oscillator(waveform=WaveformType.SINE)
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("amplitude", 0.01)  # Very quiet
        
        filt = Filter(filter_type=FilterType.LOWPASS)
        filt.set_parameter("resonance", 15.0)  # Very high Q
        
        audio_buffer = []
        duration = 4.0
        duration_samples = int(duration * 48000)
        
        for i in range(duration_samples):
            # Sweep cutoff to demonstrate self-oscillation at different frequencies
            progress = i / duration_samples
            cutoff = 200.0 + progress * 2000.0
            filt.set_parameter("cutoff_frequency", cutoff)
            
            osc_signal = osc.process()[0]
            filtered_signal = filt.process([osc_signal])[0]
            
            # Scale down to prevent clipping
            audio_buffer.append(filtered_signal.value * 0.3)
        
        filename = f"{output_dir}/filter_self_oscillation.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_multi_oscillator_filtering():
    """Demonstrate filtering of mixed oscillator signals."""
    print("Generating multi-oscillator filtering demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        # Three oscillators at different frequencies
        osc1 = Oscillator(waveform=WaveformType.SINE)
        osc1.set_parameter("frequency", 220.0)  # A3
        osc1.set_parameter("amplitude", 0.3)
        
        osc2 = Oscillator(waveform=WaveformType.SINE)
        osc2.set_parameter("frequency", 440.0)  # A4
        osc2.set_parameter("amplitude", 0.3)
        
        osc3 = Oscillator(waveform=WaveformType.SINE)
        osc3.set_parameter("frequency", 880.0)  # A5
        osc3.set_parameter("amplitude", 0.3)
        
        # Bandpass filter to isolate middle frequency
        filt = Filter(filter_type=FilterType.BANDPASS)
        filt.set_parameter("cutoff_frequency", 440.0)
        filt.set_parameter("resonance", 4.0)
        
        audio_buffer = []
        duration = 3.0
        duration_samples = int(duration * 48000)
        
        for i in range(duration_samples):
            # Mix the three oscillators
            sig1 = osc1.process()[0]
            sig2 = osc2.process()[0]
            sig3 = osc3.process()[0]
            
            # Simple mixing
            mixed_value = sig1.value + sig2.value + sig3.value
            mixed_signal = Signal(sig1.type, mixed_value)
            
            # Filter the mixed signal
            filtered_signal = filt.process([mixed_signal])[0]
            audio_buffer.append(filtered_signal.value)
        
        filename = f"{output_dir}/filter_multi_oscillator.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def main():
    """Run all filter demonstrations."""
    print("Filter Module Demonstration")
    print("=" * 40)
    
    try:
        demo_filter_types()
        demo_frequency_sweep()
        demo_resonance_sweep()
        demo_envelope_controlled_filter()
        demo_filter_oscillation()
        demo_multi_oscillator_filtering()
        
        print("\nAll demonstrations completed successfully!")
        print("Audio files generated in 'filter_examples/' directory.")
        print("\nFiles generated:")
        print("  - filter_*_demo.wav: Different filter types")
        print("  - filter_frequency_sweep.wav: Cutoff frequency sweep")
        print("  - filter_resonance_sweep.wav: Resonance sweep")
        print("  - filter_envelope_controlled.wav: Envelope modulation")
        print("  - filter_self_oscillation.wav: High resonance effects")
        print("  - filter_multi_oscillator.wav: Multi-source filtering")
        
    except Exception as e:
        import traceback
        print(f"Error during demonstration: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())