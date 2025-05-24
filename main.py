"""
Signal processing synthesizer application.

This module provides a command-line interface for generating audio signals using
various synthesizer components including oscillators, envelopes, and output modules.
"""

import numpy as np
import argparse
from signals import Oscillator, EnvelopeADSR, Mixer, OutputWav, Signal, SignalType
from signals.modules.oscillator import WaveformType

def main():
    """
    Main entry point for the signal generator application.
    
    Generates a 2-second audio file with a 440Hz sine wave modulated by an ADSR envelope.
    Optionally adds silence at the beginning based on command line arguments.
    
    The audio processing pipeline consists of:
    1. Oscillator generating a sine wave at 440Hz (A4)
    2. ADSR envelope with configurable attack, decay, sustain, and release
    3. Audio output to WAV file
    
    Command line arguments:
        --silence: Duration of silence to add at the start (in seconds)
    
    Output:
        Creates 'output_phase1.wav' file in the current directory
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a sound with optional silence at the start')
    parser.add_argument('--silence', type=float, default=0.0,
                        help='Duration of silence to add at the start (in seconds)')
    args = parser.parse_args()

    sample_rate = 48000
    duration = 2.0  # seconds
    num_samples = int(sample_rate * duration)

    # Create modules
    osc1 = Oscillator(sample_rate)
    osc1.set_parameter("frequency", 440.0) # A4
    osc1.set_parameter("waveform", "sine")

    adsr = EnvelopeADSR(sample_rate)
    adsr.set_parameter("attack", 0.020)
    adsr.set_parameter("decay", 0.8)
    adsr.set_parameter("sustain", 0.0)
    adsr.set_parameter("release", 0.1)

    # For Phase 1, we'll manually manage the output module's buffer
    # In later phases, a proper graph processor would handle this.
    output_wav = OutputWav("output_phase1.wav", sample_rate)

    # Simulate processing loop
    # Trigger envelope at the beginning and release before the end
    note_on_time = 0.0 # seconds
    note_off_time = 1.5 # seconds

    for i in range(num_samples):
        time_s = i / sample_rate

        # Envelope trigger logic
        if abs(time_s - note_on_time) < (1.0 / sample_rate):
             adsr.trigger_on()
        if abs(time_s - note_off_time) < (1.0 / sample_rate):
             adsr.trigger_off()

        osc_signal = osc1.process()[0]
        env_signal = adsr.process()[0] # Pass trigger if implementing trigger input

        modulated_signal_value = osc_signal.value * env_signal.value
        output_wav.process([Signal(SignalType.AUDIO, modulated_signal_value)])

    # Add silence at the start if requested
    if args.silence > 0:
        output_wav.add_silence_at_start(args.silence)
        print(f"Added {args.silence} seconds of silence at the start")

    output_wav.finalize()
    print(f"Phase 1 test complete. Check output_phase1.wav")

if __name__ == "__main__":
    main()
