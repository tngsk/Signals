"""
Comprehensive envelope test suite for edge cases and real-world scenarios.

This module provides extensive testing of the EnvelopeADSR module covering
edge cases, real-world musical scenarios, and stress testing that complement
the monitoring tests for complete coverage.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from signals import EnvelopeADSR, SynthContext, write_wav


@pytest.mark.unit
@pytest.mark.envelope
@pytest.mark.edge
@pytest.mark.comprehensive
class TestEnvelopeEdgeCases:
    """Test envelope behavior in edge cases and boundary conditions."""
    
    def test_zero_parameters(self):
        """Test envelope with zero-valued parameters."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            
            # Zero attack
            env.set_parameter("attack", 0.0)
            env.set_parameter("decay", 0.1)
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.1)
            
            env.trigger_on()
            signal = env.process()[0]
            # Should immediately reach peak value
            assert signal.value == 1.0, f"Zero attack should reach 1.0 immediately, got {signal.value}"
            
            # Zero decay
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)
            env.set_parameter("decay", 0.0)
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.1)
            
            env.trigger_on()
            # Process through attack
            for _ in range(int(0.01 * 48000) + 1):
                env.process()
            
            # Should be at sustain level immediately after attack
            signal = env.process()[0]
            assert abs(signal.value - 0.5) < 0.1, f"Zero decay should reach sustain quickly, got {signal.value}"
    
    def test_extreme_values(self):
        """Test envelope with extreme parameter values."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            
            # Very long times
            env.set_parameter("attack", 10.0)
            env.set_parameter("decay", 5.0)
            env.set_parameter("sustain", 0.1)
            env.set_parameter("release", 8.0)
            
            env.trigger_on()
            
            # Should still function correctly
            for _ in range(100):
                signal = env.process()[0]
                assert 0.0 <= signal.value <= 1.0, f"Value out of range: {signal.value}"
                assert not np.isnan(signal.value), "NaN detected"
                assert not np.isinf(signal.value), "Inf detected"
            
            # Very small sustain
            env.set_parameter("sustain", 0.001)
            env.trigger_off()
            
            # Should handle very small values
            for _ in range(100):
                signal = env.process()[0]
                assert signal.value >= 0.0, f"Negative value: {signal.value}"
    
    def test_rapid_parameter_changes(self):
        """Test envelope stability with rapid parameter changes."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.trigger_on()
            
            # Rapidly change parameters while processing
            for i in range(100):
                if i % 10 == 0:
                    env.set_parameter("attack", 0.01 + (i % 5) * 0.01)
                    env.set_parameter("sustain", 0.3 + (i % 7) * 0.1)
                
                signal = env.process()[0]
                assert 0.0 <= signal.value <= 1.0, f"Value out of range during rapid changes: {signal.value}"
                assert not np.isnan(signal.value), "NaN during rapid changes"
    
    def test_string_parameter_edge_cases(self):
        """Test envelope with various string parameter formats."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            
            # Test percentage formats
            test_cases = [
                ("0%", 0.0),
                ("100%", 1.0),
                ("50%", 0.5),
                ("10.5%", 0.105),
                ("0.1%", 0.001),
            ]
            
            env.set_total_duration(1.0)  # 1 second for percentage calculations
            
            for param_str, expected_ratio in test_cases:
                env.set_parameter("attack", param_str)
                expected_time = expected_ratio * 1.0  # 1 second total
                # Account for minimum time enforcement
                expected_time = max(expected_time, env.min_release_time)
                assert abs(env.attack_time - expected_time) < 0.001, f"Failed for {param_str}"
            
            # Test invalid string formats
            invalid_cases = ["abc", "50", "invalid%", ""]
            for invalid_str in invalid_cases:
                env.set_parameter("attack", invalid_str)
                # Should not crash and should use fallback value
                assert env.attack_time > 0, f"Invalid string {invalid_str} caused zero time"


@pytest.mark.integration
@pytest.mark.envelope
@pytest.mark.audio
@pytest.mark.musical
@pytest.mark.comprehensive
class TestEnvelopeMusicalScenarios:
    """Test envelope in realistic musical scenarios."""
    
    def test_piano_style_envelope(self):
        """Test envelope configured for piano-like behavior."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.005)  # Very fast attack
            env.set_parameter("decay", 0.3)     # Medium decay
            env.set_parameter("sustain", 0.2)   # Low sustain
            env.set_parameter("release", 0.8)   # Long release
            
            env.trigger_on()
            
            # Process attack phase
            attack_values = []
            for _ in range(int(0.01 * 48000)):  # 10ms
                signal = env.process()[0]
                attack_values.append(signal.value)
            
            # Should reach near peak quickly
            assert max(attack_values) > 0.9, "Piano attack should reach near peak quickly"
            
            # Continue to decay
            env.trigger_off()
            decay_values = []
            for _ in range(int(0.5 * 48000)):  # 500ms
                signal = env.process()[0]
                decay_values.append(signal.value)
            
            # Should have natural decay curve
            assert decay_values[0] > decay_values[-1], "Should decay over time"
            assert decay_values[-1] < 0.5, "Should decay significantly"
    
    def test_pad_style_envelope(self):
        """Test envelope configured for pad/string-like behavior."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.8)    # Slow attack
            env.set_parameter("decay", 0.5)     # Medium decay  
            env.set_parameter("sustain", 0.7)   # High sustain
            env.set_parameter("release", 1.2)   # Very long release
            
            env.trigger_on()
            
            # Attack should be gradual
            values = []
            for _ in range(int(0.2 * 48000)):  # 200ms
                signal = env.process()[0]
                values.append(signal.value)
            
            # Should still be rising after 200ms
            assert values[-1] < 0.9, "Pad attack should be gradual"
            assert values[-1] > values[0], "Should be increasing"
            
            # Let attack complete
            for _ in range(int(0.8 * 48000)):
                env.process()
            
            # Should reach sustain level
            signal = env.process()[0]
            assert abs(signal.value - 0.7) < 0.1, f"Should reach sustain level, got {signal.value}"
    
    def test_percussion_envelope(self):
        """Test envelope configured for percussion-like behavior."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.001)  # Instant attack
            env.set_parameter("decay", 0.05)    # Fast decay
            env.set_parameter("sustain", 0.0)   # No sustain
            env.set_parameter("release", 0.1)   # Short release
            
            env.trigger_on()
            
            # Should reach peak very quickly
            signal = env.process()[0]
            assert signal.value > 0.5, "Percussion should have fast attack"
            
            # Continue and trigger off immediately
            env.trigger_off()
            
            # Should decay quickly
            decay_values = []
            for _ in range(int(0.1 * 48000)):  # 100ms
                signal = env.process()[0]
                decay_values.append(signal.value)
            
            # Should reach very low level quickly
            assert decay_values[-1] < 0.1, "Percussion should decay to near zero"
    
    def test_envelope_retriggering(self):
        """Test envelope behavior when retriggered during different phases."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.1)
            env.set_parameter("decay", 0.1)
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.2)
            
            # Start envelope
            env.trigger_on()
            
            # Process partway through attack
            for _ in range(int(0.05 * 48000)):  # 50ms
                env.process()
            
            mid_attack_value = env.process()[0].value
            
            # Retrigger during attack
            env.trigger_on()
            restart_value = env.process()[0].value
            
            # Should restart from beginning
            assert restart_value < mid_attack_value, "Retrigger should restart envelope"
            
            # Test retrigger during sustain
            for _ in range(int(0.25 * 48000)):  # Get to sustain
                env.process()
            
            sustain_value = env.process()[0].value
            env.trigger_on()  # Retrigger
            restart_from_sustain = env.process()[0].value
            
            # Should restart from beginning
            assert restart_from_sustain < sustain_value, "Retrigger from sustain should restart"


@pytest.mark.performance
@pytest.mark.envelope
@pytest.mark.stress
@pytest.mark.comprehensive
class TestEnvelopeStressTest:
    """Stress test envelope under demanding conditions."""
    
    def test_long_duration_stability(self):
        """Test envelope stability over very long durations."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)
            env.set_parameter("decay", 0.01)
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.01)
            
            env.trigger_on()
            
            # Process for equivalent of 10 seconds
            for i in range(10 * 48000):
                signal = env.process()[0]
                
                # Check for issues every 48000 samples (1 second)
                if i % 48000 == 0:
                    assert 0.0 <= signal.value <= 1.0, f"Value out of range at sample {i}: {signal.value}"
                    assert not np.isnan(signal.value), f"NaN at sample {i}"
                    assert not np.isinf(signal.value), f"Inf at sample {i}"
                
                # Trigger release halfway through
                if i == 5 * 48000:
                    env.trigger_off()
    
    def test_rapid_triggering(self):
        """Test envelope with very rapid on/off triggering."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)
            env.set_parameter("decay", 0.01)
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.01)
            
            # Rapid triggering pattern
            for cycle in range(100):
                env.trigger_on()
                
                # Process for just a few samples
                for _ in range(10):
                    signal = env.process()[0]
                    assert 0.0 <= signal.value <= 1.0, "Value out of range during rapid triggering"
                
                env.trigger_off()
                
                # Process for a few more samples
                for _ in range(10):
                    signal = env.process()[0]
                    assert 0.0 <= signal.value <= 1.0, "Value out of range during rapid release"
    
    def test_memory_usage_stability(self):
        """Test that envelope doesn't leak memory over time."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)
            env.set_parameter("decay", 0.01)
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.01)
            
            # Process many cycles to check for memory leaks
            for cycle in range(1000):
                env.trigger_on()
                
                # Process through a complete cycle
                for _ in range(int(0.1 * 48000)):  # 100ms
                    env.process()
                
                env.trigger_off()
                
                # Process release
                for _ in range(int(0.05 * 48000)):  # 50ms
                    env.process()
                
                # Basic sanity check every 100 cycles
                if cycle % 100 == 0:
                    signal = env.process()[0]
                    assert not np.isnan(signal.value), f"NaN detected at cycle {cycle}"


@pytest.mark.audio
@pytest.mark.envelope
@pytest.mark.quality
@pytest.mark.comprehensive
class TestEnvelopeAudioQuality:
    """Test envelope audio quality and musical characteristics."""
    
    def test_envelope_smoothness_analysis(self):
        """Analyze envelope smoothness across different configurations."""
        with SynthContext(sample_rate=48000):
            test_configs = [
                {"attack": 0.01, "decay": 0.1, "sustain": 0.8, "release": 0.2},
                {"attack": 0.1, "decay": 0.2, "sustain": 0.5, "release": 0.3},
                {"attack": 0.005, "decay": 0.05, "sustain": 0.3, "release": 0.1},
            ]
            
            for config in test_configs:
                env = EnvelopeADSR()
                for param, value in config.items():
                    env.set_parameter(param, value)
                
                env.trigger_on()
                
                # Capture complete envelope
                envelope_data = []
                total_samples = int(2.0 * 48000)  # 2 seconds
                release_point = int(1.0 * 48000)  # Release at 1 second
                
                for i in range(total_samples):
                    if i == release_point:
                        env.trigger_off()
                    
                    signal = env.process()[0]
                    envelope_data.append(signal.value)
                
                # Analyze smoothness
                derivatives = np.diff(envelope_data)
                second_derivatives = np.diff(derivatives)
                
                # Check for excessive jitter
                max_second_derivative = np.max(np.abs(second_derivatives))
                assert max_second_derivative < 0.01, f"Excessive jitter in config {config}: {max_second_derivative}"
                
                # Check that envelope reaches expected levels
                max_value = np.max(envelope_data)
                assert max_value > 0.9, f"Failed to reach peak in config {config}: {max_value}"
    
    def test_envelope_frequency_response(self):
        """Test envelope doesn't introduce unwanted frequency content."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.02)
            env.set_parameter("decay", 0.1)
            env.set_parameter("sustain", 0.6)
            env.set_parameter("release", 0.15)
            
            env.trigger_on()
            
            # Generate envelope data
            envelope_data = []
            for i in range(int(1.0 * 48000)):  # 1 second
                if i == int(0.7 * 48000):  # Release at 0.7 seconds
                    env.trigger_off()
                
                signal = env.process()[0]
                envelope_data.append(signal.value)
            
            # Analyze frequency content using FFT
            fft = np.fft.fft(envelope_data)
            freqs = np.fft.fftfreq(len(envelope_data), 1/48000)
            magnitude = np.abs(fft)
            
            # Find energy above 1kHz (should be minimal for good envelope)
            high_freq_indices = np.where(np.abs(freqs) > 1000)[0]
            high_freq_energy = np.sum(magnitude[high_freq_indices])
            total_energy = np.sum(magnitude)
            
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # High frequency content should be minimal (less than 5%)
            assert high_freq_ratio < 0.05, f"Too much high frequency content: {high_freq_ratio*100:.1f}%"
    
    def test_envelope_with_audio_modulation(self, temp_dir):
        """Test envelope modulating actual audio content."""
        with SynthContext(sample_rate=48000):
            # Generate test audio (sine wave)
            duration = 1.0
            num_samples = int(duration * 48000)
            t = np.arange(num_samples) / 48000
            test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            # Apply envelope
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.05)
            env.set_parameter("decay", 0.1)
            env.set_parameter("sustain", 0.7)
            env.set_parameter("release", 0.2)
            
            env.trigger_on()
            
            modulated_audio = np.zeros(num_samples)
            release_point = int(0.7 * num_samples)
            
            for i in range(num_samples):
                if i == release_point:
                    env.trigger_off()
                
                env_signal = env.process()[0]
                modulated_audio[i] = test_audio[i] * env_signal.value
            
            # Analyze modulated audio quality
            # Check for clicks (large discontinuities)
            audio_derivatives = np.diff(modulated_audio)
            max_derivative = np.max(np.abs(audio_derivatives))
            
            # Should not have audible clicks
            assert max_derivative < 0.1, f"Audio discontinuity detected: {max_derivative}"
            
            # Check that modulation is effective
            early_rms = np.sqrt(np.mean(modulated_audio[:1000] ** 2))
            late_rms = np.sqrt(np.mean(modulated_audio[-1000:] ** 2))
            
            # Late portion should be quieter due to envelope
            assert late_rms < early_rms, "Envelope should reduce amplitude over time"
            
            # Save test audio for manual verification
            test_file = temp_dir / "envelope_modulated_test.wav"
            write_wav(str(test_file), modulated_audio, 48000)
            assert test_file.exists(), "Test audio file should be created"


@pytest.mark.regression
@pytest.mark.envelope
@pytest.mark.edge
@pytest.mark.comprehensive
class TestEnvelopeRegressionScenarios:
    """Test specific regression scenarios and bug fixes."""
    
    def test_phase_transition_precision(self):
        """Test precise phase transitions to prevent timing bugs."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.set_parameter("attack", 0.01)   # 480 samples
            env.set_parameter("decay", 0.01)    # 480 samples
            env.set_parameter("sustain", 0.5)
            env.set_parameter("release", 0.01)  # 480 samples (or more due to anti-click)
            
            env.trigger_on()
            
            # Track phase transitions
            phases = []
            for i in range(2000):  # More than enough to cover all phases
                signal = env.process()[0]
                phases.append(env._phase)
                
                # Trigger release after attack and decay
                if i == 1000:
                    env.trigger_off()
            
            # Verify phase progression
            assert 1 in phases, "Should enter attack phase"
            assert 2 in phases, "Should enter decay phase"
            assert 3 in phases, "Should enter sustain phase"
            assert 4 in phases, "Should enter release phase"
            assert 0 in phases[1500:], "Should return to idle phase"
    
    def test_parameter_update_during_processing(self):
        """Test parameter updates don't cause glitches during processing."""
        with SynthContext(sample_rate=48000):
            env = EnvelopeADSR()
            env.trigger_on()
            
            values = []
            for i in range(1000):
                # Change parameters during processing
                if i == 100:
                    env.set_parameter("attack", 0.05)
                if i == 200:
                    env.set_parameter("decay", 0.08)
                if i == 300:
                    env.set_parameter("sustain", 0.3)
                if i == 400:
                    env.set_parameter("release", 0.15)
                
                signal = env.process()[0]
                values.append(signal.value)
                
                # Verify no invalid values
                assert 0.0 <= signal.value <= 1.0, f"Invalid value at sample {i}: {signal.value}"
                assert not np.isnan(signal.value), f"NaN at sample {i}"
            
            # Check for excessive jumps
            derivatives = np.diff(values)
            max_jump = np.max(np.abs(derivatives))
            
            # Should not have excessive discontinuities
            assert max_jump < 0.5, f"Excessive jump during parameter change: {max_jump}"
    
    def test_edge_case_sample_rates(self):
        """Test envelope at various sample rates to prevent rate-dependent bugs."""
        sample_rates = [8000, 22050, 44100, 48000, 96000, 192000]
        
        for sample_rate in sample_rates:
            with SynthContext(sample_rate=sample_rate):
                env = EnvelopeADSR()
                env.set_parameter("attack", 0.01)
                env.set_parameter("decay", 0.02)
                env.set_parameter("sustain", 0.6)
                env.set_parameter("release", 0.03)
                
                env.trigger_on()
                
                # Process enough samples to cover all phases
                total_samples = int(0.1 * sample_rate)  # 100ms
                release_point = int(0.05 * sample_rate)  # 50ms
                
                for i in range(total_samples):
                    if i == release_point:
                        env.trigger_off()
                    
                    signal = env.process()[0]
                    
                    # Basic sanity checks
                    assert 0.0 <= signal.value <= 1.0, f"Invalid value at {sample_rate}Hz: {signal.value}"
                    assert not np.isnan(signal.value), f"NaN at {sample_rate}Hz"
                    assert not np.isinf(signal.value), f"Inf at {sample_rate}Hz"
                
                # Verify timing accuracy scales with sample rate
                expected_attack_samples = max(int(0.01 * sample_rate), int(env.min_release_time * sample_rate))
                actual_attack_samples = env._attack_samples
                
                # Should be close to expected (within 1 sample)
                assert abs(actual_attack_samples - expected_attack_samples) <= 1, \
                    f"Attack timing error at {sample_rate}Hz: expected {expected_attack_samples}, got {actual_attack_samples}"