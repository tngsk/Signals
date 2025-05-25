"""
Integration tests for LFO module with other synthesizer components.

This test suite validates the LFO module's integration with oscillators,
envelopes, filters, and the patch system to ensure proper modulation chains.
"""

import math
import pytest
import numpy as np
from signals import (
    LFO, Oscillator, EnvelopeADSR, VCA, Filter, Mixer,
    Signal, SignalType, WaveformType, FilterType,
    SynthContext
)


@pytest.mark.integration
class TestLFOIntegration:
    """Integration tests for LFO module with other components."""

    def test_lfo_tremolo_effect(self):
        """Test LFO creating tremolo effect with VCA."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SINE)
            lfo = LFO(waveform=WaveformType.SINE)
            vca = VCA()

            osc.set_parameter("frequency", 440.0)
            osc.set_parameter("amplitude", 0.8)
            lfo.set_parameter("frequency", 4.0)  # 4 Hz tremolo
            lfo.set_parameter("amplitude", 0.5)

            # Process tremolo effect
            results = []
            for _ in range(200):
                audio_signal = osc.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Convert LFO output to 0-1 range for VCA
                control_value = (lfo_signal.value + 1.0) * 0.5
                control_signal = Signal(SignalType.CONTROL, control_value)
                
                tremolo_signal = vca.process([audio_signal, control_signal])[0]
                results.append(tremolo_signal.value)

            # Validate tremolo effect
            assert len(results) == 200
            assert all(isinstance(x, (int, float)) for x in results)
            # Should see amplitude modulation
            rms_values = []
            for i in range(0, len(results), 20):
                chunk = results[i:i+20]
                rms = np.sqrt(np.mean([x**2 for x in chunk]))
                rms_values.append(rms)
            
            # RMS should vary due to tremolo
            assert max(rms_values) > min(rms_values) * 1.2

    def test_lfo_vibrato_effect(self):
        """Test LFO creating vibrato effect on oscillator frequency."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SINE)
            lfo = LFO(waveform=WaveformType.SINE)

            base_freq = 440.0
            osc.set_parameter("amplitude", 0.8)
            lfo.set_parameter("frequency", 20.0)  # 20 Hz vibrato (faster for testing)
            lfo.set_parameter("amplitude", 1.0)  # Full LFO range for modulation
            lfo.set_parameter("phase_offset", 0.0)  # Start at zero

            # Process vibrato effect - need enough samples to see full LFO cycle
            results = []
            frequencies_used = []
            for _ in range(2400):  # Half second at 48kHz to see multiple cycles
                lfo_signal = lfo.process()[0]
                
                # Apply frequency modulation (±5% deviation)
                vibrato_freq = base_freq * (1.0 + lfo_signal.value * 0.05)
                osc.set_parameter("frequency", vibrato_freq)
                frequencies_used.append(vibrato_freq)
                
                audio_signal = osc.process()[0]
                results.append(audio_signal.value)

            # Validate vibrato effect
            assert len(results) == 2400
            assert len(frequencies_used) == 2400
            
            # Frequency should vary around base frequency
            # With ±5% modulation, range should be roughly 418-462 Hz
            frequency_range = max(frequencies_used) - min(frequencies_used)
            assert frequency_range > 40.0  # Should have significant variation
            
            # Check that we get both positive and negative modulation
            assert min(frequencies_used) < base_freq
            assert max(frequencies_used) > base_freq

    def test_lfo_filter_cutoff_modulation(self):
        """Test LFO modulating filter cutoff frequency."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SAW)
            lfo = LFO(waveform=WaveformType.TRIANGLE)
            filt = Filter(filter_type=FilterType.LOWPASS)

            osc.set_parameter("frequency", 220.0)
            osc.set_parameter("amplitude", 0.8)
            lfo.set_parameter("frequency", 2.0)  # Faster sweep for more variation
            lfo.set_parameter("amplitude", 1.0)
            filt.set_parameter("resonance", 2.0)

            # Process filter modulation
            results = []
            cutoff_frequencies = []
            for _ in range(300):
                audio_signal = osc.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Map LFO to cutoff frequency (200Hz to 3000Hz)
                cutoff = 200.0 + (lfo_signal.value + 1.0) * 0.5 * 2800.0
                filt.set_parameter("cutoff_frequency", cutoff)
                cutoff_frequencies.append(cutoff)
                
                filtered_signal = filt.process([audio_signal])[0]
                results.append(filtered_signal.value)

            # Validate filter modulation
            assert len(results) == 300
            assert all(isinstance(x, (int, float)) for x in results)
            
            # Cutoff should vary significantly
            assert max(cutoff_frequencies) - min(cutoff_frequencies) > 2000.0

    def test_lfo_envelope_interaction(self):
        """Test LFO working with envelope simultaneously."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SQUARE)
            env = EnvelopeADSR()
            lfo = LFO(waveform=WaveformType.SINE)
            vca = VCA()

            # Configure modules
            osc.set_parameter("frequency", 330.0)
            env.set_parameter("attack", 0.1)
            env.set_parameter("decay", 0.2)
            env.set_parameter("sustain", 0.6)
            env.set_parameter("release", 0.3)
            lfo.set_parameter("frequency", 8.0)  # Fast tremolo
            lfo.set_parameter("amplitude", 0.3)

            # Trigger envelope
            env.trigger_on()

            # Process combined modulation
            results = []
            for i in range(400):
                if i == 200:  # Release halfway through
                    env.trigger_off()

                audio_signal = osc.process()[0]
                env_signal = env.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Combine envelope and LFO modulation
                combined_mod = env_signal.value * (1.0 + lfo_signal.value * 0.3)
                combined_control = Signal(SignalType.CONTROL, combined_mod)
                
                final_signal = vca.process([audio_signal, combined_control])[0]
                results.append(final_signal.value)

            # Validate combined modulation
            assert len(results) == 400
            assert not all(x == 0 for x in results)
            
            # Should show envelope shape with tremolo modulation
            # Attack phase should be louder than release phase
            attack_rms = np.sqrt(np.mean([x**2 for x in results[50:150]]))
            release_rms = np.sqrt(np.mean([x**2 for x in results[300:400]]))
            assert attack_rms > release_rms

    def test_multiple_lfo_modulation(self):
        """Test multiple LFOs modulating different parameters."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SAW)
            lfo_freq = LFO(waveform=WaveformType.SINE)  # For frequency modulation
            lfo_amp = LFO(waveform=WaveformType.TRIANGLE)  # For amplitude modulation
            vca = VCA()

            # Configure oscillator
            base_freq = 440.0
            osc.set_parameter("amplitude", 0.8)

            # Configure LFOs with different rates
            lfo_freq.set_parameter("frequency", 3.0)  # 3 Hz vibrato
            lfo_freq.set_parameter("amplitude", 0.02)  # 2% frequency deviation
            
            lfo_amp.set_parameter("frequency", 7.0)  # 7 Hz tremolo
            lfo_amp.set_parameter("amplitude", 0.4)  # 40% amplitude modulation

            # Process with dual modulation
            results = []
            for _ in range(300):
                # Frequency modulation
                freq_lfo_signal = lfo_freq.process()[0]
                modulated_freq = base_freq * (1.0 + freq_lfo_signal.value * 0.05)
                osc.set_parameter("frequency", modulated_freq)
                
                # Amplitude modulation
                amp_lfo_signal = lfo_amp.process()[0]
                control_value = (amp_lfo_signal.value + 1.0) * 0.5
                control_signal = Signal(SignalType.CONTROL, control_value)
                
                # Generate and process audio
                audio_signal = osc.process()[0]
                final_signal = vca.process([audio_signal, control_signal])[0]
                results.append(final_signal.value)

            # Validate dual modulation
            assert len(results) == 300
            assert all(isinstance(x, (int, float)) for x in results)
            
            # Should show complex modulation pattern
            # Check for variation in amplitude over time
            segment_rms = []
            for i in range(0, len(results), 30):
                chunk = results[i:i+30]
                if chunk:
                    rms = np.sqrt(np.mean([x**2 for x in chunk]))
                    segment_rms.append(rms)
            
            assert len(segment_rms) > 1
            assert max(segment_rms) > min(segment_rms) * 1.3  # Significant variation

    def test_lfo_mixer_modulation(self):
        """Test LFO modulating mixer channel gains."""
        with SynthContext(sample_rate=48000):
            osc1 = Oscillator(waveform=WaveformType.SINE)
            osc2 = Oscillator(waveform=WaveformType.SQUARE)
            lfo = LFO(waveform=WaveformType.SINE)
            mixer = Mixer(num_inputs=2)

            # Configure oscillators
            osc1.set_parameter("frequency", 440.0)
            osc1.set_parameter("amplitude", 0.6)
            osc2.set_parameter("frequency", 880.0)
            osc2.set_parameter("amplitude", 0.6)

            # Configure LFO for crossfading
            lfo.set_parameter("frequency", 1.0)  # 1 Hz crossfade
            lfo.set_parameter("amplitude", 1.0)

            # Process with gain modulation
            results = []
            for _ in range(200):
                sig1 = osc1.process()[0]
                sig2 = osc2.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Use LFO to crossfade between oscillators
                gain1 = (1.0 + lfo_signal.value) * 0.5  # 0 to 1
                gain2 = (1.0 - lfo_signal.value) * 0.5  # 1 to 0
                
                mixer.set_parameter("gain_0", gain1)
                mixer.set_parameter("gain_1", gain2)
                
                mixed_signal = mixer.process([sig1, sig2])[0]
                results.append(mixed_signal.value)

            # Validate crossfade modulation
            assert len(results) == 200
            assert all(isinstance(x, (int, float)) for x in results)
            assert not all(x == 0 for x in results)

    def test_lfo_envelope_trigger_sync(self):
        """Test LFO and envelope synchronized by common trigger."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            lfo = LFO(waveform=WaveformType.SAW)
            osc = Oscillator(waveform=WaveformType.SINE)
            vca = VCA()

            # Configure envelope
            env.set_parameter("attack", 0.05)
            env.set_parameter("decay", 0.1)
            env.set_parameter("sustain", 0.8)
            env.set_parameter("release", 0.2)

            # Configure LFO
            lfo.set_parameter("frequency", 10.0)
            lfo.set_parameter("amplitude", 0.5)

            # Configure oscillator
            osc.set_parameter("frequency", 440.0)

            # Trigger both simultaneously
            env.trigger_on()
            trigger_signal = Signal(SignalType.TRIGGER, 1.0)
            lfo.process([trigger_signal])  # Reset LFO phase

            # Process synchronized
            results = []
            for i in range(300):
                if i == 150:  # Release
                    env.trigger_off()

                audio_signal = osc.process()[0]
                env_signal = env.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Apply both envelope and LFO modulation
                modulated_amplitude = env_signal.value * (1.0 + lfo_signal.value * 0.2)
                control_signal = Signal(SignalType.CONTROL, modulated_amplitude)
                
                final_signal = vca.process([audio_signal, control_signal])[0]
                results.append(final_signal.value)

            # Validate synchronized operation
            assert len(results) == 300
            assert not all(x == 0 for x in results)
            
            # Should show envelope shape with LFO modulation
            # Should show envelope behavior
            # Check that sustain phase has signal and release phase decays
            sustain_max = max(abs(x) for x in results[100:150])
            release_max = max(abs(x) for x in results[250:300])
            assert sustain_max > release_max

    def test_lfo_parameter_automation(self):
        """Test LFO with automated parameter changes."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SINE)
            osc = Oscillator(waveform=WaveformType.SAW)

            osc.set_parameter("frequency", 220.0)
            lfo.set_parameter("amplitude", 0.8)

            results = []
            frequencies_used = []

            # Automate LFO frequency over time
            for i in range(400):
                # LFO frequency sweeps from 0.5 to 8 Hz
                lfo_freq = 0.5 + (i / 400.0) * 7.5
                lfo.set_parameter("frequency", lfo_freq)
                
                lfo_signal = lfo.process()[0]
                
                # Use LFO for frequency modulation
                base_freq = 220.0
                modulated_freq = base_freq * (1.0 + lfo_signal.value * 0.1)
                osc.set_parameter("frequency", modulated_freq)
                frequencies_used.append(modulated_freq)
                
                audio_signal = osc.process()[0]
                results.append(audio_signal.value)

            # Validate parameter automation
            assert len(results) == 400
            assert len(frequencies_used) == 400
            
            # Should see increasing modulation rate over time
            early_freq_range = max(frequencies_used[0:100]) - min(frequencies_used[0:100])
            late_freq_range = max(frequencies_used[300:400]) - min(frequencies_used[300:400])
            assert late_freq_range > early_freq_range  # Faster modulation later

    def test_lfo_performance_in_complex_patch(self):
        """Test LFO performance in complex synthesis patch."""
        with SynthContext(sample_rate=48000):
            # Create complex patch with multiple modules
            osc1 = Oscillator(waveform=WaveformType.SAW)
            osc2 = Oscillator(waveform=WaveformType.SQUARE)
            lfo1 = LFO(waveform=WaveformType.SINE)
            lfo2 = LFO(waveform=WaveformType.TRIANGLE)
            lfo3 = LFO(waveform=WaveformType.SAW)
            env = EnvelopeADSR()
            filt = Filter(filter_type=FilterType.LOWPASS)
            vca1 = VCA()
            vca2 = VCA()
            mixer = Mixer(num_inputs=2)

            # Configure all modules
            osc1.set_parameter("frequency", 110.0)
            osc2.set_parameter("frequency", 220.0)
            
            lfo1.set_parameter("frequency", 2.0)  # Filter modulation
            lfo2.set_parameter("frequency", 5.0)  # Amplitude modulation
            lfo3.set_parameter("frequency", 0.3)  # Slow crossfade
            
            env.set_parameter("attack", 0.1)
            env.set_parameter("decay", 0.3)
            env.set_parameter("sustain", 0.7)
            env.set_parameter("release", 0.5)
            env.trigger_on()
            
            filt.set_parameter("resonance", 3.0)

            import time
            start_time = time.time()

            # Process complex patch
            results = []
            for i in range(1000):  # Extended processing
                if i == 500:
                    env.trigger_off()

                # Generate audio
                sig1 = osc1.process()[0]
                sig2 = osc2.process()[0]
                
                # LFO modulations
                lfo1_sig = lfo1.process()[0]
                lfo2_sig = lfo2.process()[0]
                lfo3_sig = lfo3.process()[0]
                
                # Apply filter modulation
                cutoff = 300.0 + (lfo1_sig.value + 1.0) * 0.5 * 2000.0
                filt.set_parameter("cutoff_frequency", cutoff)
                
                # Apply amplitude modulation
                amp_mod = (lfo2_sig.value + 1.0) * 0.5
                amp_control = Signal(SignalType.CONTROL, amp_mod)
                
                # Apply crossfade modulation
                crossfade = (lfo3_sig.value + 1.0) * 0.5
                mixer.set_parameter("gain_0", crossfade)
                mixer.set_parameter("gain_1", 1.0 - crossfade)
                
                # Process signal chain
                filtered1 = filt.process([sig1])[0]
                env_sig = env.process()[0]
                
                modulated1 = vca1.process([filtered1, amp_control])[0]
                modulated2 = vca2.process([sig2, amp_control])[0]
                
                mixed = mixer.process([modulated1, modulated2])[0]
                
                # Final envelope application
                env_control = Signal(SignalType.CONTROL, env_sig.value)
                final = vca2.process([mixed, env_control])[0]
                
                results.append(final.value)

            end_time = time.time()
            processing_time = end_time - start_time

            # Validate complex patch performance
            assert len(results) == 1000
            assert all(isinstance(x, (int, float)) for x in results)
            assert not all(x == 0 for x in results)
            
            # Should process in reasonable time
            assert processing_time < 1.0  # Less than 1 second for complex patch
            
            # Should show envelope behavior
            early_rms = np.sqrt(np.mean([x**2 for x in results[100:200]]))
            late_rms = np.sqrt(np.mean([x**2 for x in results[800:900]]))
            assert early_rms > late_rms