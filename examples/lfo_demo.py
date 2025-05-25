#!/usr/bin/env python3
"""
LFO Module Demonstration

This example demonstrates the LFO (Low Frequency Oscillator) module capabilities including:
- Different LFO waveforms and their characteristics
- Tremolo effect (amplitude modulation)
- Vibrato effect (frequency modulation)
- Filter cutoff modulation
- Multiple LFO modulation
- LFO synchronization with triggers

The example generates several audio files showcasing different LFO modulation effects.
"""

import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signals import (
    SynthContext, Oscillator, LFO, EnvelopeADSR, VCA, Filter, Mixer,
    WaveformType, FilterType, Signal, SignalType, write_wav
)


def create_output_dir():
    """Create output directory for audio files."""
    output_dir = "lfo_examples"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def demo_lfo_waveforms():
    """Demonstrate different LFO waveforms modulating oscillator amplitude."""
    print("Generating LFO waveform comparison...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SINE)
        osc.set_parameter("frequency", 440.0)  # A4
        osc.set_parameter("amplitude", 0.8)
        
        lfo_waveforms = [
            (WaveformType.SINE, "sine"),
            (WaveformType.SQUARE, "square"),
            (WaveformType.TRIANGLE, "triangle"),
            (WaveformType.SAW, "saw")
        ]
        
        for waveform_type, name in lfo_waveforms:
            lfo = LFO(waveform=waveform_type)
            lfo.set_parameter("frequency", 3.0)  # 3 Hz modulation
            lfo.set_parameter("amplitude", 0.7)
            
            vca = VCA()
            
            # Generate audio with LFO modulation
            audio_buffer = []
            duration_samples = int(3.0 * 48000)  # 3 seconds
            
            for _ in range(duration_samples):
                audio_signal = osc.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Convert LFO to 0-1 range for VCA
                control_value = (lfo_signal.value + 1.0) * 0.5
                control_signal = Signal(SignalType.CONTROL, control_value)
                
                modulated_signal = vca.process([audio_signal, control_signal])[0]
                audio_buffer.append(modulated_signal.value)
            
            # Write to file
            filename = f"{output_dir}/lfo_tremolo_{name}.wav"
            write_wav(filename, np.array(audio_buffer), sample_rate=48000)
            print(f"  Generated: {filename}")


def demo_vibrato_effect():
    """Demonstrate LFO creating vibrato effect."""
    print("Generating vibrato effect demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SINE)
        lfo = LFO(waveform=WaveformType.SINE)
        
        base_freq = 440.0
        osc.set_parameter("amplitude", 0.8)
        lfo.set_parameter("frequency", 5.0)  # 5 Hz vibrato
        lfo.set_parameter("amplitude", 0.05)  # 5% frequency deviation
        
        audio_buffer = []
        duration = 4.0  # 4 seconds
        duration_samples = int(duration * 48000)
        
        for _ in range(duration_samples):
            lfo_signal = lfo.process()[0]
            
            # Apply frequency modulation
            vibrato_freq = base_freq * (1.0 + lfo_signal.value * 0.1)
            osc.set_parameter("frequency", vibrato_freq)
            
            audio_signal = osc.process()[0]
            audio_buffer.append(audio_signal.value)
        
        filename = f"{output_dir}/lfo_vibrato.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_filter_modulation():
    """Demonstrate LFO modulating filter cutoff frequency."""
    print("Generating filter cutoff modulation demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SAW)
        lfo = LFO(waveform=WaveformType.TRIANGLE)
        filt = Filter(filter_type=FilterType.LOWPASS)
        
        osc.set_parameter("frequency", 110.0)  # Low A
        osc.set_parameter("amplitude", 0.8)
        lfo.set_parameter("frequency", 0.3)  # Slow sweep
        lfo.set_parameter("amplitude", 1.0)
        filt.set_parameter("resonance", 4.0)  # High resonance for dramatic effect
        
        audio_buffer = []
        duration = 8.0  # 8 seconds
        duration_samples = int(duration * 48000)
        
        for _ in range(duration_samples):
            audio_signal = osc.process()[0]
            lfo_signal = lfo.process()[0]
            
            # Map LFO to cutoff frequency (200Hz to 3000Hz)
            cutoff = 200.0 + (lfo_signal.value + 1.0) * 0.5 * 2800.0
            filt.set_parameter("cutoff_frequency", cutoff)
            
            filtered_signal = filt.process([audio_signal])[0]
            audio_buffer.append(filtered_signal.value)
        
        filename = f"{output_dir}/lfo_filter_sweep.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_multiple_lfo_modulation():
    """Demonstrate multiple LFOs modulating different parameters."""
    print("Generating multiple LFO modulation demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SAW)
        lfo_freq = LFO(waveform=WaveformType.SINE)  # For frequency modulation
        lfo_amp = LFO(waveform=WaveformType.TRIANGLE)  # For amplitude modulation
        lfo_filter = LFO(waveform=WaveformType.SAW)  # For filter modulation
        filt = Filter(filter_type=FilterType.LOWPASS)
        vca = VCA()
        
        # Configure oscillator
        base_freq = 220.0
        osc.set_parameter("amplitude", 0.8)
        
        # Configure LFOs with different rates
        lfo_freq.set_parameter("frequency", 4.5)  # Vibrato
        lfo_freq.set_parameter("amplitude", 0.03)  # 3% frequency deviation
        
        lfo_amp.set_parameter("frequency", 6.0)  # Tremolo
        lfo_amp.set_parameter("amplitude", 0.4)  # 40% amplitude modulation
        
        lfo_filter.set_parameter("frequency", 0.2)  # Slow filter sweep
        lfo_filter.set_parameter("amplitude", 1.0)
        
        # Configure filter
        filt.set_parameter("resonance", 3.0)
        
        audio_buffer = []
        duration = 10.0  # 10 seconds
        duration_samples = int(duration * 48000)
        
        for _ in range(duration_samples):
            # Frequency modulation (vibrato)
            freq_lfo_signal = lfo_freq.process()[0]
            modulated_freq = base_freq * (1.0 + freq_lfo_signal.value * 0.08)
            osc.set_parameter("frequency", modulated_freq)
            
            # Filter modulation
            filter_lfo_signal = lfo_filter.process()[0]
            cutoff = 300.0 + (filter_lfo_signal.value + 1.0) * 0.5 * 2000.0
            filt.set_parameter("cutoff_frequency", cutoff)
            
            # Amplitude modulation (tremolo)
            amp_lfo_signal = lfo_amp.process()[0]
            control_value = (amp_lfo_signal.value + 1.0) * 0.5
            control_signal = Signal(SignalType.CONTROL, control_value)
            
            # Process audio chain
            audio_signal = osc.process()[0]
            filtered_signal = filt.process([audio_signal])[0]
            final_signal = vca.process([filtered_signal, control_signal])[0]
            audio_buffer.append(final_signal.value)
        
        filename = f"{output_dir}/lfo_multiple_modulation.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_lfo_envelope_sync():
    """Demonstrate LFO synchronized with envelope."""
    print("Generating LFO and envelope synchronization demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.SQUARE)
        env = EnvelopeADSR()
        lfo = LFO(waveform=WaveformType.SINE)
        vca = VCA()
        
        # Configure modules
        osc.set_parameter("frequency", 330.0)  # E4
        osc.set_parameter("amplitude", 0.8)
        
        env.set_parameter("attack", 0.1)
        env.set_parameter("decay", 0.3)
        env.set_parameter("sustain", 0.6)
        env.set_parameter("release", 0.8)
        
        lfo.set_parameter("frequency", 12.0)  # Fast tremolo
        lfo.set_parameter("amplitude", 0.5)
        
        audio_buffer = []
        duration = 5.0
        duration_samples = int(duration * 48000)
        note_length = int(1.2 * 48000)  # 1.2 seconds per note
        
        for i in range(duration_samples):
            # Trigger new note every 1.2 seconds
            if i % note_length == 0:
                env.trigger_on()
                # Reset LFO phase for sync
                trigger_signal = Signal(SignalType.TRIGGER, 1.0)
                lfo.process([trigger_signal])
            
            # Release note after 0.8 seconds
            if i % note_length == int(0.8 * 48000):
                env.trigger_off()
            
            # Process audio chain
            audio_signal = osc.process()[0]
            env_signal = env.process()[0]
            lfo_signal = lfo.process()[0]
            
            # Combine envelope and LFO modulation
            combined_mod = env_signal.value * (1.0 + lfo_signal.value * 0.3)
            control_signal = Signal(SignalType.CONTROL, combined_mod)
            
            final_signal = vca.process([audio_signal, control_signal])[0]
            audio_buffer.append(final_signal.value)
        
        filename = f"{output_dir}/lfo_envelope_sync.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_lfo_crossfade():
    """Demonstrate LFO creating crossfade between oscillators."""
    print("Generating LFO crossfade demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc1 = Oscillator(waveform=WaveformType.SINE)
        osc2 = Oscillator(waveform=WaveformType.SAW)
        lfo = LFO(waveform=WaveformType.SINE)
        mixer = Mixer(num_inputs=2)
        
        # Configure oscillators with harmonically related frequencies
        osc1.set_parameter("frequency", 220.0)  # A3
        osc1.set_parameter("amplitude", 0.7)
        osc2.set_parameter("frequency", 330.0)  # E4 (perfect fifth)
        osc2.set_parameter("amplitude", 0.7)
        
        # Configure LFO for slow crossfade
        lfo.set_parameter("frequency", 0.5)  # 0.5 Hz (2 second cycle)
        lfo.set_parameter("amplitude", 1.0)
        
        audio_buffer = []
        duration = 6.0  # 6 seconds
        duration_samples = int(duration * 48000)
        
        for _ in range(duration_samples):
            sig1 = osc1.process()[0]
            sig2 = osc2.process()[0]
            lfo_signal = lfo.process()[0]
            
            # Use LFO to crossfade between oscillators
            # LFO range -1 to +1, map to gains 0-1 and 1-0
            gain1 = (1.0 + lfo_signal.value) * 0.5  # 0 to 1
            gain2 = (1.0 - lfo_signal.value) * 0.5  # 1 to 0
            
            mixer.set_parameter("gain_0", gain1)
            mixer.set_parameter("gain_1", gain2)
            
            mixed_signal = mixer.process([sig1, sig2])[0]
            audio_buffer.append(mixed_signal.value)
        
        filename = f"{output_dir}/lfo_crossfade.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def demo_lfo_frequency_sweep():
    """Demonstrate LFO frequency changing over time."""
    print("Generating LFO frequency sweep demonstration...")
    
    output_dir = create_output_dir()
    
    with SynthContext(sample_rate=48000):
        osc = Oscillator(waveform=WaveformType.TRIANGLE)
        lfo = LFO(waveform=WaveformType.SINE)
        vca = VCA()
        
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("amplitude", 0.8)
        lfo.set_parameter("amplitude", 0.6)
        
        audio_buffer = []
        duration = 8.0
        duration_samples = int(duration * 48000)
        
        for i in range(duration_samples):
            # LFO frequency sweeps from 0.5 Hz to 20 Hz
            progress = i / duration_samples
            lfo_freq = 0.5 + progress * 19.5
            lfo.set_parameter("frequency", lfo_freq)
            
            # Apply tremolo
            audio_signal = osc.process()[0]
            lfo_signal = lfo.process()[0]
            
            control_value = (lfo_signal.value + 1.0) * 0.5
            control_signal = Signal(SignalType.CONTROL, control_value)
            
            modulated_signal = vca.process([audio_signal, control_signal])[0]
            audio_buffer.append(modulated_signal.value)
        
        filename = f"{output_dir}/lfo_frequency_sweep.wav"
        write_wav(filename, np.array(audio_buffer), sample_rate=48000)
        print(f"  Generated: {filename}")


def main():
    """Run all LFO demonstrations."""
    print("LFO Module Demonstration")
    print("=" * 40)
    
    try:
        demo_lfo_waveforms()
        demo_vibrato_effect()
        demo_filter_modulation()
        demo_multiple_lfo_modulation()
        demo_lfo_envelope_sync()
        demo_lfo_crossfade()
        demo_lfo_frequency_sweep()
        
        print("\nAll demonstrations completed successfully!")
        print("Audio files generated in 'lfo_examples/' directory.")
        print("\nFiles generated:")
        print("  - lfo_tremolo_*.wav: Different LFO waveforms for tremolo")
        print("  - lfo_vibrato.wav: Frequency modulation (vibrato)")
        print("  - lfo_filter_sweep.wav: Filter cutoff modulation")
        print("  - lfo_multiple_modulation.wav: Multiple LFO modulation")
        print("  - lfo_envelope_sync.wav: LFO synchronized with envelope")
        print("  - lfo_crossfade.wav: LFO crossfading between oscillators")
        print("  - lfo_frequency_sweep.wav: LFO frequency sweep")
        
    except Exception as e:
        import traceback
        print(f"Error during demonstration: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())