import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path for development if run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.modules.rust_oscillator import RustOscillator
from signals.modules.envelope import EnvelopeADSR
from signals.modules.output import OutputWav
from signals.core.module import Signal, SignalType

def main():
    parser = argparse.ArgumentParser(description="Pure Rust Analog Oscillator Demo")
    parser.add_argument("--output", type=str, default="../audio/rust_analog.wav", help="Output WAV file path")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate in Hz")
    parser.add_argument("--freq", type=float, default=440.0, help="Frequency in Hz")
    parser.add_argument("--mode", type=int, default=2, help="Oscillator mode (0=noise, 1=sine, 2=saw, 3=triangle, 4=square, 5=pulse)")

    args = parser.parse_args()

    # Ensure audio directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Rust Oscillator (mode={args.mode}) and ADSR Envelope...")
    osc = RustOscillator(sample_rate=args.sample_rate, mode=args.mode)

    env = EnvelopeADSR(sample_rate=args.sample_rate)
    env.set_parameter("attack", 0.1)
    env.set_parameter("decay", 0.2)
    env.set_parameter("sustain", 0.7)
    env.set_parameter("release", 0.5)

    out = OutputWav(filename=str(output_path), sample_rate=args.sample_rate)

    block_size = 64
    total_blocks = int((args.duration * args.sample_rate) / block_size)

    # Gate signal for the envelope
    # We'll trigger it ON at the start, and OFF for the release time duration
    release_blocks = int((0.5 * args.sample_rate) / block_size)

    print(f"Generating {args.duration}s of audio...")
    for i in range(total_blocks):
        # Gate is 1.0 until we hit the release phase
        gate_val = 1.0 if i < (total_blocks - release_blocks) else 0.0
        gate_signal = Signal(SignalType.TRIGGER, np.array([gate_val] * block_size))

        freq_signal = Signal(SignalType.AUDIO, np.array([args.freq] * block_size))
        osc_outputs = osc.process([freq_signal])

        # osc_outputs[0].value is an array of size `block_size`
        osc_block = osc_outputs[0].value

        # Process envelope and output sample by sample since legacy modules might not support block natively yet
        for j in range(block_size):
            env_out = env.process([Signal(SignalType.TRIGGER, gate_val)])
            env_val = env_out[0].value

            final_sample = osc_block[j] * env_val
            out.process([Signal(SignalType.AUDIO, final_sample)])

    out.finalize()
    print(f"Audio successfully written to {output_path}")

if __name__ == "__main__":
    main()
