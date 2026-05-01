"""
Comprehensive tests for all Signals synthesizer modules.

This test suite provides thorough testing of individual modules including
oscillators, envelopes, VCA, mixer, and output modules with various
parameter combinations and edge cases.
"""

import math
import pytest
import numpy as np
from signals import Oscillator, EnvelopeADSR, Mixer, VCA, OutputWav, LFO, Signal, SignalType, WaveformType


@pytest.mark.unit
class TestOscillator:
    """Comprehensive tests for the Oscillator module."""
    
    def test_oscillator_initialization(self, sample_rates):
        """Test oscillator initialization with different sample rates."""
        for sample_rate in sample_rates:
            osc = Oscillator(sample_rate)
            assert osc.sample_rate == sample_rate
            assert osc.input_count == 1
            assert osc.output_count == 1
            assert osc.frequency == 440.0
            assert osc.amplitude == 1.0
            assert osc.phase == 0.0
    
    def test_oscillator_waveforms(self, oscillator_module, waveform_types):
        """Test all waveform types produce valid output."""
        for waveform in waveform_types:
            osc = oscillator_module(frequency=440.0, waveform=waveform, amplitude=0.8)
            
            outputs = []
            for _ in range(100):
                output = osc.process()
                assert len(output) == 1
                assert output[0].type == SignalType.AUDIO
                assert isinstance(output[0].value, float)
                assert -1.0 <= output[0].value <= 1.0
                outputs.append(output[0].value)
            
            # Noise should have high variance, others should be periodic
            if waveform == "noise":
                assert np.std(outputs) > 0.1
            else:
                # Non-noise waveforms should have some variation
                # Square wave may only have 2 distinct values, so be more lenient
                unique_values = len(set(np.round(outputs, 6)))
                if waveform == "square":
                    assert unique_values >= 2  # Square wave has at least 2 values
                else:
                    assert unique_values > 5  # Other waveforms should have more variation
    
    def test_frequency_parameter(self, oscillator_module, test_frequencies):
        """Test frequency parameter changes."""
        osc = oscillator_module(waveform="sine")
        
        for frequency in test_frequencies:
            osc.set_parameter("frequency", frequency)
            assert osc.frequency == frequency
            
            # Generate samples and verify frequency affects output
            output = osc.process()
            assert output[0].type == SignalType.AUDIO
    
    def test_amplitude_parameter(self, oscillator_module):
        """Test amplitude parameter scaling."""
        osc = oscillator_module(waveform="sine", frequency=440.0)
        
        amplitudes = [0.0, 0.5, 1.0, 2.0]
        for amplitude in amplitudes:
            osc.set_parameter("amplitude", amplitude)
            assert osc.amplitude == amplitude
            
            # Generate samples and check scaling
            outputs = []
            for _ in range(50):
                output = osc.process()
                outputs.append(abs(output[0].value))
            
            max_output = max(outputs)
            if amplitude == 0.0:
                assert max_output == 0.0
            else:
                assert max_output <= amplitude * 1.1  # Allow small tolerance
    
    def test_waveform_switching(self, oscillator_module, waveform_types):
        """Test dynamic waveform switching."""
        osc = oscillator_module(frequency=440.0)
        
        for waveform in waveform_types:
            osc.set_parameter("waveform", waveform)
            
            # Generate samples with new waveform
            outputs = []
            for _ in range(20):
                output = osc.process()
                outputs.append(output[0].value)
            
            # Verify output is valid
            assert all(-1.0 <= x <= 1.0 for x in outputs)
    
    def test_phase_continuity(self, oscillator_module):
        """Test phase continuity across parameter changes."""
        osc = oscillator_module(frequency=440.0, waveform="sine")
        
        # Generate initial samples
        initial_outputs = []
        for _ in range(10):
            output = osc.process()
            initial_outputs.append(output[0].value)
        
        # Change amplitude (should not affect phase)
        osc.set_parameter("amplitude", 0.5)
        
        # Generate more samples
        scaled_outputs = []
        for _ in range(10):
            output = osc.process()
            scaled_outputs.append(output[0].value)
        
        # Phase should continue smoothly
        assert len(scaled_outputs) == 10
    
    def test_invalid_parameters(self, oscillator_module):
        """Test handling of invalid parameters."""
        osc = oscillator_module()
        
        # Invalid waveform should be handled gracefully
        osc.set_parameter("waveform", "invalid_waveform")
        output = osc.process()
        assert len(output) == 1  # Should still produce output
        
        # Invalid parameter names should be handled
        osc.set_parameter("invalid_param", 123)
        output = osc.process()
        assert len(output) == 1


@pytest.mark.unit
class TestEnvelopeADSR:
    """Comprehensive tests for the ADSR Envelope module."""
    
    
    def test_envelope_parameters(self, envelope_module):
        """Test envelope parameter changes."""
        env = envelope_module()
        
        # Test attack parameter
        env.set_parameter("attack", 0.05)
        assert env.attack_time == 0.05
        
        # Test decay parameter
        env.set_parameter("decay", 0.2)
        assert env.decay_time == 0.2
        
        # Test sustain parameter
        env.set_parameter("sustain", 0.8)
        assert env.sustain_level == 0.8
        
        # Test release parameter
        env.set_parameter("release", 0.3)
        assert env.release_time == 0.3
    
    def test_envelope_relative_parameters(self, envelope_module):
        """Test relative parameter formats."""
        env = envelope_module()
        env.set_total_duration(2.0)
        
        # Test percentage parameters
        env.set_parameter("attack", "10%")
        assert abs(env.attack_time - 0.2) < 0.01  # 10% of 2.0s
        
        env.set_parameter("decay", "25%")
        assert abs(env.decay_time - 0.5) < 0.01  # 25% of 2.0s
        
        # Test auto parameter
        env.set_parameter("release", "auto")
        assert env.release_time > 0
    
    def test_envelope_trigger_signals(self, envelope_module):
        """Test envelope response to trigger signals."""
        env = envelope_module()
        
        # Test trigger on via signal
        trigger_on = Signal(SignalType.TRIGGER, 1.0)
        output = env.process([trigger_on])
        assert output[0].type == SignalType.CONTROL
        
        # Process a few samples
        for _ in range(5):
            output = env.process()
        
        # Test trigger off via signal
        trigger_off = Signal(SignalType.TRIGGER, 0.0)
        output = env.process([trigger_off])
        assert output[0].type == SignalType.CONTROL
    
    def test_envelope_edge_cases(self, envelope_module):
        """Test envelope edge cases."""
        env = envelope_module()
        
        # Zero attack time
        env.set_parameter("attack", 0.0)
        env.trigger_on()
        output = env.process()
        assert 0 <= output[0].value <= 1
        
        # Zero release time
        env.set_parameter("release", 0.0)
        env.trigger_off()
        output = env.process()
        assert output[0].value >= 0
        
        # Sustain level of 0
        env.set_parameter("sustain", 0.0)
        assert env.sustain_level == 0.0
        
        # Sustain level of 1
        env.set_parameter("sustain", 1.0)
        assert env.sustain_level == 1.0


@pytest.mark.unit
class TestVCA:
    """Comprehensive tests for the VCA module."""
    
    def test_vca_initialization(self, vca_module, sample_rates):
        """Test VCA initialization."""
        for sample_rate in sample_rates:
            vca = VCA(sample_rate)
            assert vca.sample_rate == sample_rate
            assert vca.input_count == 2
            assert vca.output_count == 1
            assert vca.gain == 1.0
    
    def test_vca_basic_modulation(self, vca_module):
        """Test basic VCA amplitude modulation."""
        vca = vca_module()
        
        # Test with audio and control signals
        audio_signal = Signal(SignalType.AUDIO, 0.5)
        control_signal = Signal(SignalType.CONTROL, 0.8)
        
        output = vca.process([audio_signal, control_signal])
        assert len(output) == 1
        assert output[0].type == SignalType.AUDIO
        
        # Output should be audio * control * gain
        expected = 0.5 * 0.8 * 1.0
        assert abs(output[0].value - expected) < 1e-6
    
    def test_vca_gain_parameter(self, vca_module):
        """Test VCA gain parameter."""
        vca = vca_module()
        
        gain_values = [0.0, 0.5, 1.0, 2.0]
        for gain in gain_values:
            vca.set_parameter("gain", gain)
            assert vca.gain == gain
            
            # Test with signals
            audio = Signal(SignalType.AUDIO, 0.5)
            control = Signal(SignalType.CONTROL, 0.8)
            
            output = vca.process([audio, control])
            expected = 0.5 * 0.8 * gain
            assert abs(output[0].value - expected) < 1e-6
    
    def test_vca_signal_types(self, vca_module):
        """Test VCA with different signal type combinations."""
        vca = vca_module()
        
        # Audio + Control (normal case)
        audio = Signal(SignalType.AUDIO, 0.5)
        control = Signal(SignalType.CONTROL, 0.8)
        output = vca.process([audio, control])
        assert output[0].value == 0.5 * 0.8 * 1.0
        
        # Audio + Audio (should work)
        audio1 = Signal(SignalType.AUDIO, 0.5)
        audio2 = Signal(SignalType.AUDIO, 0.6)
        output = vca.process([audio1, audio2])
        assert output[0].value == 0.5 * 0.6 * 1.0
        
        # Wrong signal types (should output silence)
        trigger = Signal(SignalType.TRIGGER, 1.0)
        output = vca.process([trigger, control])
        assert output[0].value == 0.0
    
    def test_vca_missing_inputs(self, vca_module):
        """Test VCA with missing inputs."""
        vca = vca_module()
        
        # No inputs
        output = vca.process([])
        assert output[0].value == 0.0
        
        # Only one input (audio only)
        audio = Signal(SignalType.AUDIO, 0.5)
        output = vca.process([audio])
        assert output[0].value == 0.5 * 1.0  # audio * gain
        
        # Only one input (non-audio)
        control = Signal(SignalType.CONTROL, 0.8)
        output = vca.process([control])
        assert output[0].value == 0.0
    
    def test_vca_extreme_values(self, vca_module):
        """Test VCA with extreme input values."""
        vca = vca_module()
        
        # Large values
        audio = Signal(SignalType.AUDIO, 10.0)
        control = Signal(SignalType.CONTROL, 5.0)
        output = vca.process([audio, control])
        assert output[0].value == 50.0
        
        # Negative values
        audio = Signal(SignalType.AUDIO, -0.5)
        control = Signal(SignalType.CONTROL, 0.8)
        output = vca.process([audio, control])
        assert output[0].value == -0.4
        
        # Zero values
        audio = Signal(SignalType.AUDIO, 0.0)
        control = Signal(SignalType.CONTROL, 1.0)
        output = vca.process([audio, control])
        assert output[0].value == 0.0


@pytest.mark.unit
class TestMixer:
    """Comprehensive tests for the Mixer module."""
    
    def test_mixer_initialization(self, mixer_module):
        """Test mixer initialization with different input counts."""
        input_counts = [1, 2, 4, 8, 16]
        for num_inputs in input_counts:
            mixer = Mixer(num_inputs)
            assert mixer.input_count == num_inputs
            assert mixer.output_count == 1
            assert len(mixer.gains) == num_inputs
            assert all(gain == 1.0 for gain in mixer.gains)
    
    def test_mixer_basic_mixing(self, mixer_module):
        """Test basic audio mixing functionality."""
        mixer = mixer_module(num_inputs=3)
        
        inputs = [
            Signal(SignalType.AUDIO, 0.2),
            Signal(SignalType.AUDIO, 0.3),
            Signal(SignalType.AUDIO, 0.5)
        ]
        
        output = mixer.process(inputs)
        assert len(output) == 1
        assert output[0].type == SignalType.AUDIO
        
        # Sum should be 0.2 + 0.3 + 0.5 = 1.0
        assert abs(output[0].value - 1.0) < 1e-6
    
    def test_mixer_gain_controls(self, mixer_module):
        """Test individual channel gain controls."""
        mixer = mixer_module(num_inputs=3)
        
        # Set different gains
        mixer.set_parameter("gain1", 0.5)
        mixer.set_parameter("gain2", 2.0)
        mixer.set_parameter("gain3", 0.0)
        
        inputs = [
            Signal(SignalType.AUDIO, 1.0),
            Signal(SignalType.AUDIO, 1.0),
            Signal(SignalType.AUDIO, 1.0)
        ]
        
        output = mixer.process(inputs)
        # Expected: 1.0*0.5 + 1.0*2.0 + 1.0*0.0 = 2.5
        assert abs(output[0].value - 2.5) < 1e-6
    
    def test_mixer_signal_types(self, mixer_module):
        """Test mixer with different signal types."""
        mixer = mixer_module(num_inputs=3)
        
        inputs = [
            Signal(SignalType.AUDIO, 0.5),
            Signal(SignalType.CONTROL, 0.3),  # Should be ignored
            Signal(SignalType.AUDIO, 0.2)
        ]
        
        output = mixer.process(inputs)
        # Only audio signals should be mixed: 0.5 + 0.2 = 0.7
        assert abs(output[0].value - 0.7) < 1e-6
    
    def test_mixer_partial_inputs(self, mixer_module):
        """Test mixer with fewer inputs than channels."""
        mixer = mixer_module(num_inputs=4)
        
        # Only provide 2 inputs for 4-channel mixer
        inputs = [
            Signal(SignalType.AUDIO, 0.3),
            Signal(SignalType.AUDIO, 0.7)
        ]
        
        output = mixer.process(inputs)
        assert abs(output[0].value - 1.0) < 1e-6
    
    def test_mixer_no_inputs(self, mixer_module):
        """Test mixer with no inputs."""
        mixer = mixer_module(num_inputs=2)
        
        output = mixer.process([])
        assert output[0].value == 0.0
        
        output = mixer.process(None)
        assert output[0].value == 0.0
    
    def test_mixer_negative_gains(self, mixer_module):
        """Test mixer with negative gain values."""
        mixer = mixer_module(num_inputs=2)
        
        mixer.set_parameter("gain1", -1.0)
        mixer.set_parameter("gain2", 0.5)
        
        inputs = [
            Signal(SignalType.AUDIO, 0.8),
            Signal(SignalType.AUDIO, 0.6)
        ]
        
        output = mixer.process(inputs)
        # Expected: 0.8*(-1.0) + 0.6*0.5 = -0.8 + 0.3 = -0.5
        assert abs(output[0].value - (-0.5)) < 1e-6
    
    def test_mixer_gain_parameter_validation(self, mixer_module):
        """Test mixer gain parameter validation."""
        mixer = mixer_module(num_inputs=3)
        
        # Valid gain parameters
        mixer.set_parameter("gain1", 1.5)
        mixer.set_parameter("gain2", 0.0)
        mixer.set_parameter("gain3", -0.5)
        
        assert mixer.gains[0] == 1.5
        assert mixer.gains[1] == 0.0
        assert mixer.gains[2] == -0.5
        
        # Invalid gain index (should be handled gracefully)
        mixer.set_parameter("gain5", 1.0)  # Only 3 channels
        assert len(mixer.gains) == 3
        
        # Invalid parameter name
        mixer.set_parameter("invalid_param", 1.0)
        assert len(mixer.gains) == 3


@pytest.mark.unit
class TestOutputWav:
    """Comprehensive tests for the OutputWav module."""
    
    def test_output_initialization(self, temp_dir, sample_rates):
        """Test OutputWav initialization."""
        for sample_rate in sample_rates:
            filename = temp_dir / f"test_{sample_rate}.wav"
            output = OutputWav(str(filename), sample_rate)
            
            assert output.filename == str(filename)
            assert output.sample_rate == sample_rate
            assert output.input_count == 1
            assert output.output_count == 0
            assert output.bits_per_sample == 16
    
    def test_output_bit_depths(self, temp_dir):
        """Test OutputWav with different bit depths."""
        bit_depths = [16, 24, 32]
        for bits in bit_depths:
            filename = temp_dir / f"test_{bits}bit.wav"
            output = OutputWav(str(filename), 48000, bits)
            assert output.bits_per_sample == bits
    
    def test_output_processing(self, temp_dir):
        """Test OutputWav audio processing."""
        filename = temp_dir / "test_processing.wav"
        output = OutputWav(str(filename), 48000)
        
        # Process audio signals
        for i in range(100):
            audio_signal = Signal(SignalType.AUDIO, 0.5 * np.sin(2 * np.pi * 440 * i / 48000))
            result = output.process([audio_signal])
            
            # Output module should return empty list
            assert result == []
        
        # Finalize should create file
        output.finalize()
        assert filename.exists()
        assert filename.stat().st_size > 0
    
    def test_output_non_audio_signals(self, temp_dir):
        """Test OutputWav with non-audio signals."""
        filename = temp_dir / "test_non_audio.wav"
        output = OutputWav(str(filename), 48000)
        
        # Process non-audio signals (should be ignored)
        control_signal = Signal(SignalType.CONTROL, 0.5)
        trigger_signal = Signal(SignalType.TRIGGER, 1.0)
        
        output.process([control_signal])
        output.process([trigger_signal])
        
        # No audio data should be captured
        output.finalize()
        # File should still be created but minimal size
    
    def test_output_empty_buffer(self, temp_dir):
        """Test OutputWav with empty buffer."""
        filename = temp_dir / "test_empty.wav"
        output = OutputWav(str(filename), 48000)
        
        # Finalize without adding any audio
        output.finalize()
        
        # File should not be created for empty buffer
        # (or should be created with minimal content)


@pytest.mark.integration
class TestModuleChaining:
    """Integration tests for chaining multiple modules together."""
    
    def test_oscillator_vca_chain(self, oscillator_module, vca_module):
        """Test chaining oscillator with VCA."""
        osc = oscillator_module(frequency=440.0, waveform="sine", amplitude=0.8)
        vca = vca_module(gain=1.0)
        
        # Chain processing
        for _ in range(50):
            osc_output = osc.process()
            control_signal = Signal(SignalType.CONTROL, 0.5)
            
            vca_output = vca.process([osc_output[0], control_signal])
            
            assert vca_output[0].type == SignalType.AUDIO
            assert abs(vca_output[0].value) <= 0.8 * 0.5  # amplitude * control
    
    def test_envelope_vca_modulation(self, envelope_module, vca_module):
        """Test envelope modulating VCA."""
        env = envelope_module(attack=0.01, decay=0.1, sustain=0.5, release=0.1)
        vca = vca_module()
        
        env.trigger_on()
        
        # Test modulation over time
        audio_signal = Signal(SignalType.AUDIO, 1.0)
        
        results = []
        for _ in range(100):
            env_output = env.process()
            vca_output = vca.process([audio_signal, env_output[0]])
            results.append(vca_output[0].value)
        
        # Should see envelope shape in VCA output
        assert len(results) == 100
        assert all(0 <= x <= 1 for x in results)
    
    def test_multiple_oscillators_mixer(self, oscillator_module, mixer_module):
        """Test multiple oscillators mixed together."""
        osc1 = oscillator_module(frequency=440.0, waveform="sine")
        osc2 = oscillator_module(frequency=880.0, waveform="square")
        osc3 = oscillator_module(frequency=220.0, waveform="saw")
        
        mixer = mixer_module(num_inputs=3, gain1=0.5, gain2=0.3, gain3=0.7)
        
        # Process mixed signal
        for _ in range(20):
            out1 = osc1.process()[0]
            out2 = osc2.process()[0]
            out3 = osc3.process()[0]
            
            mixed = mixer.process([out1, out2, out3])
            
            assert mixed[0].type == SignalType.AUDIO
            # Mixed signal should be within reasonable range
            assert abs(mixed[0].value) <= 3.0  # Sum of max amplitudes * gains
    


@pytest.mark.unit
class TestLFO:
    """Comprehensive tests for the LFO module."""
    
    def test_lfo_initialization(self, sample_rates):
        """Test LFO initialization with different sample rates."""
        for sample_rate in sample_rates:
            lfo = LFO(sample_rate)
            assert lfo.sample_rate == sample_rate
            assert lfo.input_count == 1  # Optional trigger input
            assert lfo.output_count == 1
            assert lfo.frequency == 1.0
            assert lfo.amplitude == 1.0
            assert lfo.phase_offset == 0.0
            assert lfo.phase == 0.0
    
    
    def test_lfo_frequency_parameter(self, lfo_module):
        """Test LFO frequency parameter control."""
        lfo = lfo_module()
        
        # Test frequency changes
        lfo.set_parameter("frequency", 2.0)
        assert lfo.frequency == 2.0
        
        lfo.set_parameter("frequency", 0.5)
        assert lfo.frequency == 0.5
        
        # Test frequency limits
        lfo.set_parameter("frequency", 0.001)  # Below minimum
        assert lfo.frequency == 0.01  # Should be clamped
        
        lfo.set_parameter("frequency", 100.0)  # Above maximum
        assert lfo.frequency == 50.0  # Should be clamped
    
    def test_lfo_amplitude_parameter(self, lfo_module):
        """Test LFO amplitude parameter control."""
        lfo = lfo_module()
        
        # Test amplitude changes
        lfo.set_parameter("amplitude", 0.5)
        assert lfo.amplitude == 0.5
        
        # Test amplitude scaling in output
        lfo.set_parameter("frequency", 10.0)  # Fast for quick test
        outputs = []
        for _ in range(50):
            output = lfo.process()[0]
            outputs.append(abs(output.value))
        
        # Maximum output should be close to amplitude
        assert max(outputs) <= lfo.amplitude + 0.1
        
        # Test amplitude limits
        lfo.set_parameter("amplitude", -0.5)
        assert lfo.amplitude == 0.0  # Should be clamped
        
        lfo.set_parameter("amplitude", 2.0)
        assert lfo.amplitude == 1.0  # Should be clamped
    
    def test_lfo_phase_offset(self, lfo_module):
        """Test LFO phase offset parameter."""
        lfo1 = lfo_module()
        lfo2 = lfo_module()
        
        lfo1.set_parameter("frequency", 5.0)
        lfo2.set_parameter("frequency", 5.0)
        lfo2.set_parameter("phase_offset", 90.0)  # 90 degree offset
        
        # Get initial outputs
        out1 = lfo1.process()[0].value
        out2 = lfo2.process()[0].value
        
        # They should be different due to phase offset
        assert abs(out1 - out2) > 0.1
        
        # Test phase offset wrapping
        lfo2.set_parameter("phase_offset", 450.0)  # Should become 90.0
        assert lfo2.phase_offset == 90.0
    
    def test_lfo_waveform_parameter(self, lfo_module):
        """Test LFO waveform parameter changes."""
        lfo = lfo_module()
        
        # Test waveform change by enum
        lfo.set_parameter("waveform", WaveformType.SQUARE)
        assert lfo.waveform == WaveformType.SQUARE
        
        # Test waveform change by string
        lfo.set_parameter("waveform", "triangle")
        assert lfo.waveform == WaveformType.TRIANGLE
    
    def test_lfo_trigger_input(self, lfo_module):
        """Test LFO trigger input for phase reset."""
        lfo = lfo_module()
        lfo.set_parameter("frequency", 2.0)
        
        # Let LFO run for a while
        for _ in range(50):
            lfo.process()
        
        initial_phase = lfo.phase
        assert initial_phase > 0.0
        
        # Send trigger signal
        trigger = Signal(SignalType.TRIGGER, 1.0)
        lfo.process([trigger])
        
        # Phase should be reset
        assert abs(lfo.phase) < 0.01
    
    def test_lfo_invalid_parameters(self, lfo_module):
        """Test LFO handling of invalid parameters."""
        lfo = lfo_module()
        original_freq = lfo.frequency
        
        # Test invalid parameter name
        lfo.set_parameter("invalid_param", 5.0)
        assert lfo.frequency == original_freq
        
        # Test invalid waveform
        original_waveform = lfo.waveform
        lfo.set_parameter("waveform", "invalid")
        assert lfo.waveform == original_waveform
    
    def test_lfo_reset(self, lfo_module):
        """Test LFO reset functionality."""
        lfo = lfo_module()
        lfo.set_parameter("phase_offset", 45.0)
        
        # Let LFO run
        for _ in range(50):
            lfo.process()
        
        # Reset LFO
        lfo.reset()
        
        # Phase should be reset to phase offset
        expected_phase = 45.0 / 360.0
        assert abs(lfo.phase - expected_phase) < 0.01
    
    def test_lfo_modulation_integration(self, lfo_module, oscillator_module, vca_module):
        """Test LFO integration with other modules for modulation."""
        osc = oscillator_module(frequency=440.0, amplitude=0.8)
        lfo = lfo_module()
        vca = vca_module()
        
        lfo.set_parameter("frequency", 4.0)  # 4 Hz tremolo
        lfo.set_parameter("amplitude", 0.5)
        
        # Process tremolo effect
        results = []
        for _ in range(100):
            audio_signal = osc.process()[0]
            lfo_signal = lfo.process()[0]
            
            # Convert LFO to 0-1 range for VCA
            control_value = (lfo_signal.value + 1.0) * 0.5
            control_signal = Signal(SignalType.CONTROL, control_value)
            
            tremolo_signal = vca.process([audio_signal, control_signal])[0]
            results.append(tremolo_signal.value)
        
        # Should see amplitude modulation
        assert len(results) == 100
        rms_values = []
        for i in range(0, len(results), 10):
            chunk = results[i:i+10]
            rms = np.sqrt(np.mean([x**2 for x in chunk]))
            rms_values.append(rms)
        
        # RMS should vary due to tremolo
        assert len(rms_values) > 1
        assert max(rms_values) > min(rms_values) * 1.1



class TestFilter:
    """Test cases for Filter module functionality."""

    def test_filter_initialization(self):
        """Test Filter module initialization with default parameters."""
        from signals.modules.filter import Filter, FilterType
        
        filt = Filter(sample_rate=48000)
        assert filt.sample_rate == 48000
        assert filt.filter_type == FilterType.LOWPASS
        assert filt.cutoff_frequency == 1000.0
        assert filt.resonance == 0.707
        assert filt.input_count == 1
        assert filt.output_count == 1

