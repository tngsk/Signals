use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use rnbo_analog_rust::{AdsrEnvelope, AnalogOscillator};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "../audio/rnbo_analog_rust.wav")]
    output: String,

    #[arg(long, default_value_t = 2.0)]
    duration: f32,

    #[arg(long, default_value_t = 48000.0)]
    sample_rate: f32,

    #[arg(long, default_value_t = 440.0)]
    freq: f32,

    #[arg(long, default_value_t = 2)]
    mode: u32,
}

fn main() {
    let args = Args::parse();

    let mut osc = AnalogOscillator::new(args.sample_rate, args.mode);
    let mut env = AdsrEnvelope::new(args.sample_rate, 0.1, 0.2, 0.7, 0.5);

    let spec = WavSpec {
        channels: 1,
        sample_rate: args.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    // Ensure output directory exists
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    }

    let mut writer = WavWriter::create(&args.output, spec).expect("Failed to create WAV writer");

    let block_size = 64;
    let total_samples = (args.duration * args.sample_rate) as usize;
    let total_blocks = total_samples / block_size;
    let release_samples = (0.5 * args.sample_rate) as usize;
    let release_block_idx = total_blocks - (release_samples / block_size);

    let freq_block = vec![args.freq; block_size];
    let mut out_block = vec![0.0; block_size];

    env.trigger_on();

    for i in 0..total_blocks {
        if i == release_block_idx {
            env.trigger_off();
        }

        osc.process_block(&freq_block, &mut out_block);

        for j in 0..block_size {
            let env_val = env.step();
            let sample = out_block[j] * env_val;

            // Convert to 16-bit PCM
            let pcm_sample = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(pcm_sample).unwrap();
        }
    }

    writer.finalize().expect("Failed to finalize WAV file");
    println!("Audio successfully written to {}", args.output);
}
