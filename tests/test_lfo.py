"""
Comprehensive tests for LFO (Low Frequency Oscillator) module.

This test suite validates the LFO module functionality including:
- Basic LFO generation and waveform types
- Parameter control (frequency, amplitude, phase offset)
- Trigger input and phase reset functionality
- Integration with other modules for modulation
- Performance and stability characteristics
"""

import math
import pytest
import numpy as np
from signals import (
    LFO, Oscillator, VCA, EnvelopeADSR,
    Signal, SignalType, WaveformType,
    SynthContext
)


class TestLFOBasic:
    """Basic LFO functionality tests."""

    def test_lfo_initialization(self):
        """Test LFO initialization with default parameters."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            assert lfo.sample_rate == 48000
            assert lfo.waveform == WaveformType.SINE
            assert lfo.frequency == 1.0
            assert lfo.amplitude == 1.0
            assert lfo.phase_offset == 0.0
            assert lfo.phase == 0.0

    def test_lfo_initialization_with_parameters(self):
        """Test LFO initialization with custom parameters."""
        with SynthContext(sample_rate=44100):
            lfo = LFO(waveform=WaveformType.TRIANGLE)
            
            assert lfo.sample_rate == 44100
            assert lfo.waveform == WaveformType.TRIANGLE

    def test_lfo_explicit_sample_rate(self):
        """Test LFO with explicit sample rate."""
        lfo = LFO(sample_rate=96000, waveform=WaveformType.SQUARE)
        
        assert lfo.sample_rate == 96000
        assert lfo.waveform == WaveformType.SQUARE

    def test_lfo_input_output_count(self):
        """Test LFO input/output configuration."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            assert lfo.input_count == 1  # Optional trigger input
            assert lfo.output_count == 1  # Control signal output


class TestLFOWaveforms:
    """Test LFO waveform generation."""

    def test_lfo_sine_wave(self):
        """Test LFO sine wave generation."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SINE)
            lfo.set_parameter("frequency", 1.0)  # 1 Hz
            
            # Generate one complete cycle (48000 samples = 1 second at 1 Hz)
            samples = []
            for _ in range(48000):
                signal = lfo.process()[0]
                assert signal.type == SignalType.CONTROL
                samples.append(signal.value)
            
            # Check sine wave properties
            assert len(samples) == 48000
            assert abs(max(samples) - 1.0) < 0.01  # Peak near 1.0
            assert abs(min(samples) - (-1.0)) < 0.01  # Trough near -1.0
            
            # Check key points in sine wave cycle
            assert abs(samples[0]) < 0.1  # Start near zero
            assert abs(samples[12000] - 1.0) < 0.1  # Quarter cycle at peak
            assert abs(samples[24000]) < 0.1  # Half cycle at zero crossing
            assert abs(samples[36000] - (-1.0)) < 0.1  # Three quarter cycle at trough

    def test_lfo_square_wave(self):
        """Test LFO square wave generation."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SQUARE)
            lfo.set_parameter("frequency", 2.0)  # 2 Hz
            
            samples = []
            for _ in range(24000):  # Half second
                signal = lfo.process()[0]
                samples.append(signal.value)
            
            # Square wave should be mostly +1 or -1
            unique_values = set(samples)
            assert len(unique_values) == 2  # Only two values
            assert 1.0 in unique_values
            assert -1.0 in unique_values

    def test_lfo_triangle_wave(self):
        """Test LFO triangle wave generation."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.TRIANGLE)
            lfo.set_parameter("frequency", 1.0)
            
            samples = []
            for _ in range(48000):
                signal = lfo.process()[0]
                samples.append(signal.value)
            
            # Triangle wave should reach peaks and have linear segments
            assert abs(max(samples) - 1.0) < 0.01
            assert abs(min(samples) - (-1.0)) < 0.01

    def test_lfo_saw_wave(self):
        """Test LFO sawtooth wave generation."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SAW)
            lfo.set_parameter("frequency", 1.0)
            
            samples = []
            for _ in range(48000):
                signal = lfo.process()[0]
                samples.append(signal.value)
            
            # Sawtooth should go from -1 to +1 linearly
            assert abs(max(samples) - 1.0) < 0.01
            assert abs(min(samples) - (-1.0)) < 0.01

    def test_lfo_noise_wave(self):
        """Test LFO noise generation."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.NOISE)
            
            samples = []
            for _ in range(1000):
                signal = lfo.process()[0]
                samples.append(signal.value)
            
            # Noise should have random values within range
            assert all(-1.0 <= s <= 1.0 for s in samples)
            # Should have some variation
            assert len(set(samples)) > 10


class TestLFOParameters:
    """Test LFO parameter control."""

    def test_frequency_parameter(self):
        """Test LFO frequency parameter."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Test frequency change
            lfo.set_parameter("frequency", 2.0)
            assert lfo.frequency == 2.0
            
            # Test frequency limits
            lfo.set_parameter("frequency", 0.005)  # Below minimum
            assert lfo.frequency == 0.01  # Should be clamped
            
            lfo.set_parameter("frequency", 100.0)  # Above maximum
            assert lfo.frequency == 50.0  # Should be clamped

    def test_amplitude_parameter(self):
        """Test LFO amplitude parameter."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Test amplitude change
            lfo.set_parameter("amplitude", 0.5)
            assert lfo.amplitude == 0.5
            
            # Generate samples and check amplitude scaling
            lfo.set_parameter("frequency", 1.0)
            samples = []
            for _ in range(24000):  # Half cycle
                signal = lfo.process()[0]
                samples.append(signal.value)
            
            # Maximum should be near amplitude value
            assert abs(max(samples) - 0.5) < 0.01
            
            # Test amplitude limits
            lfo.set_parameter("amplitude", -0.5)  # Below minimum
            assert lfo.amplitude == 0.0  # Should be clamped
            
            lfo.set_parameter("amplitude", 2.0)  # Above maximum
            assert lfo.amplitude == 1.0  # Should be clamped

    def test_phase_offset_parameter(self):
        """Test LFO phase offset parameter."""
        with SynthContext(sample_rate=48000):
            lfo1 = LFO()
            lfo2 = LFO()
            
            lfo1.set_parameter("frequency", 1.0)
            lfo2.set_parameter("frequency", 1.0)
            lfo2.set_parameter("phase_offset", 90.0)  # 90 degrees offset
            
            # Generate samples from both LFOs
            samples1 = []
            samples2 = []
            for _ in range(12000):  # Quarter cycle
                s1 = lfo1.process()[0]
                s2 = lfo2.process()[0]
                samples1.append(s1.value)
                samples2.append(s2.value)
            
            # LFO2 should be ahead by 90 degrees (quarter cycle)
            # At start, LFO1 should be ~0, LFO2 should be ~1
            assert abs(samples1[0]) < 0.1  # Near zero
            assert abs(samples2[0] - 1.0) < 0.1  # Near peak

    def test_waveform_parameter(self):
        """Test LFO waveform parameter changes."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Test waveform change by enum
            lfo.set_parameter("waveform", WaveformType.SQUARE)
            assert lfo.waveform == WaveformType.SQUARE
            
            # Test waveform change by string
            lfo.set_parameter("waveform", "triangle")
            assert lfo.waveform == WaveformType.TRIANGLE
            
            # Test invalid waveform
            original_waveform = lfo.waveform
            lfo.set_parameter("waveform", "invalid")
            assert lfo.waveform == original_waveform  # Should remain unchanged

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Test invalid parameter name
            original_freq = lfo.frequency
            lfo.set_parameter("invalid_param", 5.0)
            assert lfo.frequency == original_freq  # Should remain unchanged


class TestLFOTriggerInput:
    """Test LFO trigger input functionality."""

    def test_trigger_phase_reset(self):
        """Test LFO phase reset with trigger input."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SINE)
            lfo.set_parameter("frequency", 1.0)
            
            # Let LFO run for a quarter cycle
            for _ in range(12000):
                lfo.process()
            
            # Current phase should be around 0.25
            phase_before = lfo.phase
            assert 0.2 < phase_before < 0.3
            
            # Send trigger signal
            trigger = Signal(SignalType.TRIGGER, 1.0)
            lfo.process([trigger])
            
            # Phase should be reset to 0
            assert abs(lfo.phase) < 0.01

    def test_trigger_with_phase_offset(self):
        """Test trigger reset with phase offset."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            lfo.set_parameter("phase_offset", 90.0)  # 90 degrees
            
            # Let LFO run
            for _ in range(12000):
                lfo.process()
            
            # Send trigger
            trigger = Signal(SignalType.TRIGGER, 1.0)
            lfo.process([trigger])
            
            # Phase should reset to offset position (0.25)
            assert abs(lfo.phase - 0.25) < 0.01

    def test_no_trigger_on_low_signal(self):
        """Test that low trigger signals don't reset phase."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Let LFO run
            for _ in range(12000):
                lfo.process()
            
            phase_before = lfo.phase
            expected_next_phase = (phase_before + lfo.phase_increment) % 1.0
            
            # Send low trigger signal
            low_trigger = Signal(SignalType.TRIGGER, 0.3)
            lfo.process([low_trigger])
            
            # Phase should continue normally (not reset)
            assert abs(lfo.phase - expected_next_phase) < 0.001

    def test_non_trigger_input_ignored(self):
        """Test that non-trigger inputs are ignored."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Let LFO run
            for _ in range(12000):
                lfo.process()
            
            phase_before = lfo.phase
            expected_next_phase = (phase_before + lfo.phase_increment) % 1.0
            
            # Send non-trigger signal
            audio_signal = Signal(SignalType.AUDIO, 1.0)
            lfo.process([audio_signal])
            
            # Phase should continue normally (not reset)
            assert abs(lfo.phase - expected_next_phase) < 0.001


class TestLFOModulation:
    """Test LFO modulation integration with other modules."""

    def test_lfo_vca_modulation(self):
        """Test LFO modulating VCA amplitude."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SINE)
            lfo = LFO(waveform=WaveformType.SINE)
            vca = VCA()
            
            osc.set_parameter("frequency", 440.0)
            osc.set_parameter("amplitude", 1.0)
            lfo.set_parameter("frequency", 4.0)  # 4 Hz tremolo
            lfo.set_parameter("amplitude", 0.5)  # 50% modulation depth
            
            # Process chain: OSC -> VCA <- LFO
            modulated_samples = []
            for _ in range(12000):  # Quarter second
                audio_signal = osc.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Scale LFO output for VCA (0-1 range)
                scaled_lfo = Signal(SignalType.CONTROL, (lfo_signal.value + 1.0) * 0.5)
                
                modulated_signal = vca.process([audio_signal, scaled_lfo])[0]
                modulated_samples.append(modulated_signal.value)
            
            # Should see amplitude modulation
            assert len(modulated_samples) == 12000
            assert not all(abs(s) == abs(modulated_samples[0]) for s in modulated_samples)

    def test_lfo_frequency_modulation(self):
        """Test LFO modulating oscillator frequency (vibrato)."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SINE)
            lfo = LFO(waveform=WaveformType.SINE)
            
            base_freq = 440.0
            lfo.set_parameter("frequency", 5.0)  # 5 Hz vibrato
            lfo.set_parameter("amplitude", 0.1)  # 10% frequency deviation
            
            # Apply frequency modulation
            frequencies_used = []
            for _ in range(9600):  # 0.2 seconds
                lfo_signal = lfo.process()[0]
                
                # Apply vibrato: base_freq +/- 10%
                modulated_freq = base_freq * (1.0 + lfo_signal.value * 0.1)
                osc.set_parameter("frequency", modulated_freq)
                frequencies_used.append(modulated_freq)
                
                osc.process()  # Generate audio sample
            
            # Should see frequency variation
            assert min(frequencies_used) < base_freq
            assert max(frequencies_used) > base_freq
            assert abs(max(frequencies_used) - min(frequencies_used)) > 5.0  # Significant variation

    def test_lfo_envelope_interaction(self):
        """Test LFO and envelope working together."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator(waveform=WaveformType.SAW)
            env = EnvelopeADSR()
            lfo = LFO(waveform=WaveformType.TRIANGLE)
            vca = VCA()
            
            # Setup envelope
            env.set_parameter("attack", 0.1)
            env.set_parameter("decay", 0.2)
            env.set_parameter("sustain", 0.7)
            env.set_parameter("release", 0.5)
            env.trigger_on()
            
            # Setup LFO for tremolo
            lfo.set_parameter("frequency", 6.0)
            lfo.set_parameter("amplitude", 0.3)
            
            # Process combined modulation
            combined_samples = []
            for i in range(48000):  # 1 second
                if i == 24000:  # Release after 0.5 seconds
                    env.trigger_off()
                
                audio_signal = osc.process()[0]
                env_signal = env.process()[0]
                lfo_signal = lfo.process()[0]
                
                # Combine envelope and LFO
                combined_mod = env_signal.value * (1.0 + lfo_signal.value * 0.3)
                combined_control = Signal(SignalType.CONTROL, combined_mod)
                
                final_signal = vca.process([audio_signal, combined_control])[0]
                combined_samples.append(final_signal.value)
            
            # Should show both envelope shape and tremolo
            assert len(combined_samples) == 48000
            # Early samples should show attack
            assert max(combined_samples[1000:5000]) > max(combined_samples[0:1000])
            # Later samples should show release
            assert max(combined_samples[40000:48000]) < max(combined_samples[20000:30000])


class TestLFOPerformance:
    """Test LFO performance characteristics."""

    def test_lfo_processing_speed(self):
        """Test LFO processing performance."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SINE)
            lfo.set_parameter("frequency", 2.0)
            
            import time
            start_time = time.time()
            
            # Process many samples
            for _ in range(48000):
                lfo.process()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process 1 second of audio in reasonable time
            assert processing_time < 0.1  # Less than 100ms

    def test_lfo_frequency_stability(self):
        """Test LFO frequency accuracy over time."""
        with SynthContext(sample_rate=48000):
            lfo = LFO(waveform=WaveformType.SINE)
            target_freq = 2.0
            lfo.set_parameter("frequency", target_freq)
            
            # Track zero crossings to measure actual frequency
            samples = []
            zero_crossings = []
            
            for i in range(96000):  # 2 seconds
                signal = lfo.process()[0]
                samples.append(signal.value)
                
                # Detect zero crossings (positive going)
                if i > 0 and samples[i-1] < 0 and samples[i] >= 0:
                    zero_crossings.append(i)
            
            # Should have approximately 4 zero crossings in 2 seconds at 2 Hz
            assert 2 <= len(zero_crossings) <= 6
            
            # Check timing between crossings
            if len(zero_crossings) >= 2:
                avg_period = (zero_crossings[-1] - zero_crossings[0]) / (len(zero_crossings) - 1)
                measured_freq = 48000 / avg_period  # Zero crossings occur at target frequency
                assert abs(measured_freq - target_freq) < 0.2  # Within 10% tolerance

    def test_lfo_memory_stability(self):
        """Test LFO memory usage stability."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Process extended duration
            for _ in range(480000):  # 10 seconds worth
                lfo.process()
            
            # LFO should maintain stable state
            assert 0.0 <= lfo.phase <= 1.0
            assert not math.isnan(lfo.phase)
            assert not math.isinf(lfo.phase)


class TestLFOEdgeCases:
    """Test LFO edge cases and error handling."""

    def test_very_low_frequency(self):
        """Test LFO with very low frequency."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            lfo.set_parameter("frequency", 0.01)  # 0.01 Hz (100 second period)
            
            # Should still generate reasonable output
            samples = []
            for _ in range(4800):  # 0.1 seconds
                signal = lfo.process()[0]
                samples.append(signal.value)
                assert -1.0 <= signal.value <= 1.0
            
            # Values should change very slowly
            assert abs(samples[-1] - samples[0]) < 0.1

    def test_high_frequency_lfo(self):
        """Test LFO at higher frequencies."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            lfo.set_parameter("frequency", 50.0)  # 50 Hz (upper limit)
            
            samples = []
            for _ in range(4800):  # 0.1 seconds
                signal = lfo.process()[0]
                samples.append(signal.value)
                assert -1.0 <= signal.value <= 1.0
            
            # Should complete several cycles
            assert len(set(samples)) > 10  # Should have many different values

    def test_zero_amplitude(self):
        """Test LFO with zero amplitude."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            lfo.set_parameter("amplitude", 0.0)
            
            samples = []
            for _ in range(1000):
                signal = lfo.process()[0]
                samples.append(signal.value)
            
            # All samples should be zero
            assert all(s == 0.0 for s in samples)

    def test_phase_offset_wraparound(self):
        """Test phase offset values that wrap around."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            
            # Test various wraparound values
            lfo.set_parameter("phase_offset", 450.0)  # Should become 90.0
            assert lfo.phase_offset == 90.0
            
            lfo.set_parameter("phase_offset", -90.0)  # Should become 270.0
            assert lfo.phase_offset == 270.0
            
            lfo.set_parameter("phase_offset", 720.0)  # Should become 0.0
            assert lfo.phase_offset == 0.0

    def test_reset_functionality(self):
        """Test LFO reset functionality."""
        with SynthContext(sample_rate=48000):
            lfo = LFO()
            lfo.set_parameter("phase_offset", 45.0)
            
            # Let LFO run
            for _ in range(12000):
                lfo.process()
            
            phase_before_reset = lfo.phase
            
            # Reset LFO
            lfo.reset()
            
            # Phase should be reset to phase offset
            expected_phase = 45.0 / 360.0
            assert abs(lfo.phase - expected_phase) < 0.01

    def test_get_info(self):
        """Test LFO info retrieval."""
        with SynthContext(sample_rate=44100):
            lfo = LFO(waveform=WaveformType.TRIANGLE)
            lfo.set_parameter("frequency", 3.5)
            lfo.set_parameter("amplitude", 0.8)
            lfo.set_parameter("phase_offset", 120.0)
            
            info = lfo.get_info()
            
            assert info["module_type"] == "LFO"
            assert info["waveform"] == "triangle"
            assert info["frequency"] == 3.5
            assert info["amplitude"] == 0.8
            assert info["phase_offset"] == 120.0
            assert info["sample_rate"] == 44100
            assert "current_phase" in info
            assert "phase_increment" in info