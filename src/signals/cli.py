"""
CLI entry points for standalone execution of Signals modules.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development if run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.core.module import Signal, SignalType
from signals.core.store import SignalStore
from signals.core.store_node import StoreNode
from signals.modules.filter import Filter
from signals.modules.oscillator import Oscillator
from signals.modules.output import OutputWav


def signals_osc():
    """CLI entry point for running an isolated oscillator."""
    parser = argparse.ArgumentParser(description="Signals Oscillator CLI")
    parser.add_argument("--waveform", type=str, default="sine", help="Waveform type (sine, square, triangle, saw)")
    parser.add_argument("--freq", type=float, default=440.0, help="Frequency in Hz")
    parser.add_argument("--output", type=str, default="temp.wav", help="Output WAV file path")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate in Hz")

    args = parser.parse_args()

    # We will build a small ad-hoc pipeline: Oscillator -> OutputWav
    store = SignalStore()

    osc = Oscillator(sample_rate=args.sample_rate)
    osc.set_parameter("waveform", args.waveform)
    osc.set_parameter("frequency", args.freq)
    osc_node = StoreNode("osc", osc, output_keys={0: "bus_audio"})

    out = OutputWav(args.output, sample_rate=args.sample_rate)
    out_node = StoreNode("out", out, input_keys={0: "bus_audio"})

    print(f"Generating {args.duration}s of {args.waveform} wave at {args.freq}Hz...")

    num_samples = int(args.duration * args.sample_rate)
    for _ in range(num_samples):
        osc_node.process(store)
        out_node.process(store)

    out.finalize()
    print(f"Saved to {args.output}")


def signals_filter():
    """CLI entry point for running an isolated filter."""
    parser = argparse.ArgumentParser(description="Signals Filter CLI")
    parser.add_argument("--input", type=str, required=True, help="Input WAV file path")
    parser.add_argument("--cutoff", type=float, default=1000.0, help="Cutoff frequency in Hz")
    parser.add_argument("--type", type=str, default="lowpass", help="Filter type (lowpass, highpass, bandpass)")
    parser.add_argument("--output", type=str, default="final.wav", help="Output WAV file path")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate in Hz (should match input)")

    args = parser.parse_args()

    import numpy as np
    import wave

    try:
        with wave.open(args.input, 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

            # Convert to numpy array
            if sampwidth == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
                data = data.astype(np.float32) / 32768.0
            elif sampwidth == 4:
                data = np.frombuffer(raw_data, dtype=np.int32)
                data = data.astype(np.float32) / 2147483648.0
            else:
                print(f"Unsupported sample width: {sampwidth} bytes")
                return

            # Mix down to mono if stereo
            if channels > 1:
                data = data.reshape(-1, channels)
                data = data.mean(axis=1)

    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    store = SignalStore()

    filt = Filter(sample_rate=sr)
    filt.set_parameter("filter_type", args.type)
    filt.set_parameter("cutoff_frequency", args.cutoff)
    filt_node = StoreNode("filt", filt, input_keys={0: "bus_in"}, output_keys={0: "bus_out"})

    out = OutputWav(args.output, sample_rate=sr)
    out_node = StoreNode("out", out, input_keys={0: "bus_out"})

    print(f"Filtering {args.input} with {args.type} at {args.cutoff}Hz...")

    for sample in data:
        # WavRead node equivalent
        store.set("bus_in", Signal(SignalType.AUDIO, sample))

        # Process filter and output
        filt_node.process(store)
        out_node.process(store)

    out.finalize()
    print(f"Saved to {args.output}")

def main():
    print("Use specific module commands like signals-osc or signals-filter")

if __name__ == "__main__":
    main()
