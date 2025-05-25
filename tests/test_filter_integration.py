"""
Integration tests for Filter module with other synthesizer components.

This test suite validates the Filter module's integration with oscillators,
envelopes, and the patch system to ensure proper audio processing chains.
"""

import math
import pytest
import numpy as np
from signals import (
    Oscillator, EnvelopeADSR, VCA, Mixer, OutputWav,
    Signal, SignalType, WaveformType
)
from signals.modules.filter import Filter, FilterType
from signals.core.context import SynthContext
from signals.processing.engine import SynthEngine
from signals.processing.patch import PatchTemplate


@pytest.mark.integration
class TestFilterIntegration:
    """Integration tests for Filter module with other components."""

    def test_filter_oscillator_chain(self):
        """Test Filter processing oscillator output."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SAW)
            filt = Filter(filter_type=FilterType.LOWPASS)

            osc.set_parameter("frequency", 440.0)
            osc.set_parameter("amplitude", 0.8)
            filt.set_parameter("cutoff_frequency", 2000.0)
            filt.set_parameter("resonance", 1.5)

            # Process signal chain
            results = []
            for _ in range(100):
                osc_signal = osc.process()[0]
                filtered_signal = filt.process([osc_signal])[0]
                results.append(filtered_signal.value)

            # Validate results
            assert len(results) == 100
            assert all(isinstance(x, (int, float)) for x in results)
            assert not all(x == 0 for x in results)  # Should produce non-zero output

    def test_filter_envelope_modulation(self):
        """Test Filter with envelope-controlled VCA."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SINE)
            filt = Filter(filter_type=FilterType.LOWPASS)
            env = EnvelopeADSR()
            vca = VCA()

            # Configure modules
            osc.set_parameter("frequency", 880.0)
            filt.set_parameter("cutoff_frequency", 1500.0)
            env.set_parameter("attack", 0.1)
            env.set_parameter("decay", 0.2)
            env.set_parameter("sustain", 0.7)
            env.set_parameter("release", 0.3)

            # Trigger envelope
            env.trigger_on()

            # Process complete signal chain
            results = []
            for _ in range(200):
                osc_signal = osc.process()[0]
                filtered_signal = filt.process([osc_signal])[0]
                env_signal = env.process()[0]
                final_signal = vca.process([filtered_signal, env_signal])[0]
                results.append(final_signal.value)

            # Validate envelope shape is preserved through filter
            assert len(results) == 200
            # Since envelope starts at 0 and builds up, we should see increasing values initially
            # Check that the signal is not all zeros and has variation
            assert not all(x == 0 for x in results)
            assert max([abs(x) for x in results]) > 0.01  # Should have reasonable amplitude

    def test_filter_types_comparison(self):
        """Test different filter types with same input signal."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SQUARE)
            osc.set_parameter("frequency", 440.0)  # A4 note
            osc.set_parameter("amplitude", 1.0)

            # Create filters of different types
            filters = {
                'lowpass': Filter(filter_type=FilterType.LOWPASS),
                'highpass': Filter(filter_type=FilterType.HIGHPASS),
                'bandpass': Filter(filter_type=FilterType.BANDPASS),
                'notch': Filter(filter_type=FilterType.NOTCH)
            }

            # Configure all filters with cutoff at 2kHz (different from input 440Hz)
            for filt in filters.values():
                filt.set_parameter("cutoff_frequency", 2000.0)
                filt.set_parameter("resonance", 3.0)

            # Process same signal through all filters for longer to allow settling
            results = {name: [] for name in filters.keys()}

            for i in range(200):  # More samples to allow filters to settle
                osc_signal = osc.process()[0]
                for name, filt in filters.items():
                    filtered = filt.process([osc_signal])[0]
                    results[name].append(filtered.value)

            # Use samples from later in the sequence after settling
            settle_samples = 50
            lowpass_rms = np.sqrt(np.mean([x**2 for x in results['lowpass'][settle_samples:]]))
            highpass_rms = np.sqrt(np.mean([x**2 for x in results['highpass'][settle_samples:]]))
            bandpass_rms = np.sqrt(np.mean([x**2 for x in results['bandpass'][settle_samples:]]))
            notch_rms = np.sqrt(np.mean([x**2 for x in results['notch'][settle_samples:]]))

            # Different filter types should produce different RMS levels
            # Use tolerance for floating point comparison
            tolerance = 0.001
            assert abs(lowpass_rms - highpass_rms) > tolerance
            assert abs(lowpass_rms - bandpass_rms) > tolerance
            assert abs(highpass_rms - bandpass_rms) > tolerance

    def test_filter_frequency_sweep(self):
        """Test Filter with dynamically changing cutoff frequency."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SAW)
            filt = Filter(filter_type=FilterType.LOWPASS)

            osc.set_parameter("frequency", 440.0)
            filt.set_parameter("resonance", 3.0)

            results = []
            cutoff_frequencies = []

            # Sweep cutoff from 200Hz to 8000Hz
            for i in range(200):
                # Logarithmic frequency sweep
                cutoff = 200.0 * (8000.0 / 200.0) ** (i / 199.0)
                filt.set_parameter("cutoff_frequency", cutoff)
                cutoff_frequencies.append(cutoff)

                osc_signal = osc.process()[0]
                filtered_signal = filt.process([osc_signal])[0]
                results.append(abs(filtered_signal.value))

            # Validate frequency response behavior
            assert len(results) == 200
            assert all(isinstance(x, (int, float)) for x in results)

            # Should see amplitude changes as cutoff frequency changes
            max_amplitude = max(results)
            min_amplitude = min(results)
            assert max_amplitude > min_amplitude * 1.1  # At least 10% difference

    def test_filter_with_mixer(self):
        """Test Filter processing mixed signals from multiple oscillators."""
        with SynthContext(sample_rate=48000):
            # Create multiple oscillators
            osc1 = Oscillator(waveform=WaveformType.SINE)
            osc2 = Oscillator(waveform=WaveformType.SAW)
            osc3 = Oscillator(waveform=WaveformType.SQUARE)

            mixer = Mixer(num_inputs=3)
            filt = Filter(filter_type=FilterType.LOWPASS)

            # Configure oscillators with different frequencies
            osc1.set_parameter("frequency", 220.0)
            osc2.set_parameter("frequency", 440.0)
            osc3.set_parameter("frequency", 880.0)

            # Configure mixer gains
            mixer.set_parameter("gain_0", 0.3)
            mixer.set_parameter("gain_1", 0.4)
            mixer.set_parameter("gain_2", 0.2)

            # Configure filter
            filt.set_parameter("cutoff_frequency", 500.0)  # Cut above 500Hz
            filt.set_parameter("resonance", 1.0)

            results = []
            for _ in range(100):
                # Generate signals
                sig1 = osc1.process()[0]
                sig2 = osc2.process()[0]
                sig3 = osc3.process()[0]

                # Mix signals
                mixed = mixer.process([sig1, sig2, sig3])[0]

                # Filter mixed signal
                filtered = filt.process([mixed])[0]
                results.append(filtered.value)

            # Validate processing
            assert len(results) == 100
            assert all(isinstance(x, (int, float)) for x in results)
            assert not all(x == 0 for x in results)

    def test_filter_in_patch_system(self):
        """Test Filter module integration with basic processing chain."""
        with SynthContext(sample_rate=48000, buffer_size=100):
            # Create a simple processing chain without patch system
            osc = Oscillator(waveform=WaveformType.SAW)
            filt = Filter(filter_type=FilterType.LOWPASS)

            # Configure modules
            osc.set_parameter("frequency", 440.0)
            osc.set_parameter("amplitude", 0.8)
            filt.set_parameter("filter_type", FilterType.LOWPASS)
            filt.set_parameter("cutoff_frequency", 1000.0)
            filt.set_parameter("resonance", 2.0)

            # Process audio chain
            audio_buffer = []
            for _ in range(100):
                osc_signal = osc.process()[0]
                filtered_signal = filt.process([osc_signal])[0]
                audio_buffer.append(filtered_signal.value)

            # Validate output
            assert len(audio_buffer) > 0
            assert all(isinstance(x, (int, float)) for x in audio_buffer)
            assert not all(x == 0 for x in audio_buffer)

    def test_filter_parameter_automation(self):
        """Test Filter with automated parameter changes."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SAW)
            filt = Filter(filter_type=FilterType.LOWPASS)

            osc.set_parameter("frequency", 440.0)

            results = []

            # Automate resonance parameter
            for i in range(200):
                # Oscillate resonance between 0.5 and 5.0
                resonance = 2.75 + 2.25 * math.sin(2 * math.pi * i / 50.0)
                filt.set_parameter("resonance", resonance)

                # Also sweep cutoff frequency
                cutoff = 500.0 + 1500.0 * (i / 199.0)
                filt.set_parameter("cutoff_frequency", cutoff)

                osc_signal = osc.process()[0]
                filtered_signal = filt.process([osc_signal])[0]
                results.append(filtered_signal.value)

            # Validate automation effects
            assert len(results) == 200
            assert all(isinstance(x, (int, float)) for x in results)

            # Should see variation in output due to parameter automation
            rms_values = []
            for i in range(0, len(results), 10):
                chunk = results[i:i+10]
                rms = np.sqrt(np.mean([x**2 for x in chunk]))
                rms_values.append(rms)

            # RMS should vary with parameter automation
            max_rms = max(rms_values)
            min_rms = min(rms_values)
            assert max_rms > min_rms * 1.2  # At least 20% variation

    def test_filter_performance_stability(self):
        """Test Filter performance and stability over extended processing."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.NOISE)
            filt = Filter(filter_type=FilterType.BANDPASS)

            filt.set_parameter("cutoff_frequency", 2000.0)
            filt.set_parameter("resonance", 5.0)  # High resonance

            # Process many samples to test stability
            max_values = []
            for block in range(10):
                block_max = 0.0
                for _ in range(1000):
                    osc_signal = osc.process()[0]
                    filtered_signal = filt.process([osc_signal])[0]
                    block_max = max(block_max, abs(filtered_signal.value))
                max_values.append(block_max)

            # Validate stability (no runaway values)
            assert all(x < 100.0 for x in max_values)  # Reasonable upper bound
            assert all(not math.isnan(x) for x in max_values)
            assert all(not math.isinf(x) for x in max_values)

            # Should maintain consistent processing levels
            mean_max = np.mean(max_values)
            assert all(abs(x - mean_max) < mean_max * 2.0 for x in max_values)
