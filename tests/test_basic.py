"""
Basic tests for the signals package.
"""

import pytest
import numpy as np
from signals import Oscillator, EnvelopeADSR, Mixer, OutputWav, Signal, SignalType


class TestSignal:
    """Test Signal class functionality."""
    
    def test_signal_creation(self):
        """Test Signal creation with different types and values."""
        # Test with float value
        audio_signal = Signal(SignalType.AUDIO, 0.5)
        assert audio_signal.type == SignalType.AUDIO
        assert audio_signal.value == 0.5
        
        # Test with numpy array
        control_array = np.array([0.1, 0.2, 0.3])
        control_signal = Signal(SignalType.CONTROL, control_array)
        assert control_signal.type == SignalType.CONTROL
        np.testing.assert_array_equal(control_signal.value, control_array)
    
    def test_signal_repr(self):
        """Test Signal string representation."""
        signal = Signal(SignalType.TRIGGER, 1.0)
        repr_str = repr(signal)
        assert "trigger" in repr_str.lower()
        assert "value_shape" in repr_str


class TestOscillator:
    """Test Oscillator module functionality."""
    
    def test_oscillator_creation(self):
        """Test Oscillator instantiation."""
        osc = Oscillator(sample_rate=48000)
        assert osc.input_count == 1  # Has input for frequency modulation
        assert osc.output_count == 1
    
    def test_oscillator_sine_wave(self):
        """Test sine wave generation."""
        osc = Oscillator(sample_rate=48000)
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("waveform", "sine")
        
        # Generate a few samples
        outputs = []
        for _ in range(100):
            output = osc.process()
            assert len(output) == 1
            assert output[0].type == SignalType.AUDIO
            assert isinstance(output[0].value, float)
            assert -1.0 <= output[0].value <= 1.0
            outputs.append(output[0].value)
        
        # Check that we get a varying signal (not constant)
        assert len(set(outputs)) > 1
    
    def test_oscillator_frequency_change(self):
        """Test frequency parameter changes."""
        osc = Oscillator(sample_rate=48000)
        
        # Test setting different frequencies
        frequencies = [220.0, 440.0, 880.0]
        for freq in frequencies:
            osc.set_parameter("frequency", freq)
            output = osc.process()
            assert len(output) == 1
            assert output[0].type == SignalType.AUDIO


class TestEnvelopeADSR:
    """Test ADSR Envelope functionality."""
    
    def test_envelope_creation(self):
        """Test envelope instantiation."""
        env = EnvelopeADSR(sample_rate=48000)
        assert env.input_count == 1
        assert env.output_count == 1
    
    def test_envelope_trigger_cycle(self):
        """Test envelope trigger on/off cycle."""
        env = EnvelopeADSR(sample_rate=48000)
        env.set_parameter("attack", 0.01)
        env.set_parameter("decay", 0.1)
        env.set_parameter("sustain", 0.5)
        env.set_parameter("release", 0.1)
        
        # Initial state should be zero
        output = env.process()
        assert output[0].value == 0.0
        
        # Trigger on should start attack
        env.trigger_on()
        output = env.process()
        assert output[0].type == SignalType.CONTROL
        assert 0.0 <= output[0].value <= 1.0
        
        # Process several samples during attack
        for _ in range(10):
            output = env.process()
            assert 0.0 <= output[0].value <= 1.0
        
        # Trigger off should start release
        env.trigger_off()
        release_start_value = env.process()[0].value
        
        # Process several samples during release
        for _ in range(10):
            output = env.process()
            assert 0.0 <= output[0].value <= release_start_value


class TestMixer:
    """Test Mixer functionality."""
    
    def test_mixer_creation(self):
        """Test mixer instantiation."""
        mixer = Mixer(num_inputs=4)
        assert mixer.input_count == 4
        assert mixer.output_count == 1
    
    def test_mixer_no_inputs(self):
        """Test mixer with no inputs."""
        mixer = Mixer(num_inputs=2)
        output = mixer.process([])
        assert len(output) == 1
        assert output[0].type == SignalType.AUDIO
        assert output[0].value == 0.0
    
    def test_mixer_single_input(self):
        """Test mixer with single audio input."""
        mixer = Mixer(num_inputs=2)
        input_signal = Signal(SignalType.AUDIO, 0.5)
        
        output = mixer.process([input_signal])
        assert len(output) == 1
        assert output[0].type == SignalType.AUDIO
        assert output[0].value == 0.5
    
    def test_mixer_multiple_inputs(self):
        """Test mixer with multiple audio inputs."""
        mixer = Mixer(num_inputs=3)
        inputs = [
            Signal(SignalType.AUDIO, 0.3),
            Signal(SignalType.AUDIO, 0.2),
            Signal(SignalType.AUDIO, 0.1)
        ]
        
        output = mixer.process(inputs)
        assert len(output) == 1
        assert output[0].type == SignalType.AUDIO
        assert abs(output[0].value - 0.6) < 1e-6  # 0.3 + 0.2 + 0.1
    
    def test_mixer_gain_control(self):
        """Test mixer gain parameters."""
        mixer = Mixer(num_inputs=2)
        mixer.set_parameter("gain1", 0.5)
        mixer.set_parameter("gain2", 2.0)
        
        inputs = [
            Signal(SignalType.AUDIO, 1.0),
            Signal(SignalType.AUDIO, 0.5)
        ]
        
        output = mixer.process(inputs)
        expected = 1.0 * 0.5 + 0.5 * 2.0  # 0.5 + 1.0 = 1.5
        assert abs(output[0].value - expected) < 1e-6


class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_oscillator_envelope_chain(self):
        """Test chaining oscillator with envelope."""
        osc = Oscillator(sample_rate=48000)
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("waveform", "sine")
        
        env = EnvelopeADSR(sample_rate=48000)
        env.set_parameter("attack", 0.01)
        env.trigger_on()
        
        # Process a few samples
        for _ in range(10):
            osc_output = osc.process()
            env_output = env.process()
            
            # Modulate oscillator with envelope
            modulated_value = osc_output[0].value * env_output[0].value
            assert -1.0 <= modulated_value <= 1.0
    
    def test_simple_synthesis_pipeline(self):
        """Test a simple synthesis pipeline."""
        # Create modules
        osc = Oscillator(sample_rate=48000)
        osc.set_parameter("frequency", 440.0)
        osc.set_parameter("waveform", "sine")
        
        env = EnvelopeADSR(sample_rate=48000)
        env.trigger_on()
        
        mixer = Mixer(num_inputs=1)
        
        # Process pipeline for several samples
        results = []
        for _ in range(50):
            osc_signal = osc.process()[0]
            env_signal = env.process()[0]
            
            # Manual modulation
            modulated_signal = Signal(
                SignalType.AUDIO, 
                osc_signal.value * env_signal.value
            )
            
            mixed_output = mixer.process([modulated_signal])
            results.append(mixed_output[0].value)
        
        # Verify we got meaningful output
        assert len(results) == 50
        assert not all(x == 0.0 for x in results)  # Should have non-zero values
        assert all(-1.0 <= x <= 1.0 for x in results)  # Within valid range