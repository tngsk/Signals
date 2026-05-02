use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::path::PathBuf;
use std::collections::HashMap;

use signals_core::graph::ModuleGraph;
use signals_core::module::Module;
use signals_core::rnbo_module::RNBOModule;

// For pure Rust oscillator
use rnbo_analog_rust::AnalogOscillator;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    patch: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Patch {
    name: String,
    description: Option<String>,
    sample_rate: f64,
    modules: HashMap<String, ModuleConfig>,
    connections: Vec<ConnectionConfig>,
    sequence: Vec<SequenceEvent>,
}

#[derive(Debug, Deserialize)]
struct ModuleConfig {
    #[serde(rename = "type")]
    module_type: String,
    #[serde(default)]
    parameters: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Deserialize)]
struct ConnectionConfig {
    #[serde(alias = "source")]
    from: String,
    #[serde(alias = "target")]
    to: String,
}

#[derive(Debug, Deserialize)]
struct SequenceEvent {
    time: f64,
    action: String,
    target: String,
}

struct BasicVCA {}

impl BasicVCA {
    fn new() -> Self { Self {} }
}

impl Module for BasicVCA {
    fn input_count(&self) -> usize { 2 } // audio in, cv in
    fn output_count(&self) -> usize { 1 }

    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let block_size = inputs.get(0).map_or(64, |v| v.len());
        let mut out = vec![0.0; block_size];

        if inputs.len() >= 2 && inputs[0].len() == block_size && inputs[1].len() == block_size {
            for i in 0..block_size {
                out[i] = inputs[0][i] * inputs[1][i];
            }
        } else if inputs.len() >= 1 && inputs[0].len() == block_size {
             for i in 0..block_size {
                out[i] = inputs[0][i];
            }
        }
        vec![out]
    }
}

// ADSR envelope implementation
struct ADSRModule {
    env: rnbo_analog_rust::AdsrEnvelope,
}

impl ADSRModule {
    fn new(sample_rate: f64, attack: f64, decay: f64, sustain: f64, release: f64) -> Self {
        Self {
            env: rnbo_analog_rust::AdsrEnvelope::new(sample_rate as f32, attack as f32, decay as f32, sustain as f32, release as f32),
        }
    }
}

impl Module for ADSRModule {
    fn input_count(&self) -> usize { 0 }
    fn output_count(&self) -> usize { 1 }

    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let block_size = if !inputs.is_empty() && !inputs[0].is_empty() { inputs[0].len() } else { 64 };
        let mut out = vec![0.0; block_size];
        for i in 0..block_size {
            out[i] = self.env.step() as f64;
        }
        vec![out]
    }

    fn handle_message(&mut self, msg: &str) {
        match msg {
            "trigger" | "trigger_on" => self.env.trigger_on(),
            "release" | "trigger_off" => self.env.trigger_off(),
            _ => {}
        }
    }
}

// Rust analog oscillator wrapper to match the Module trait
struct AnalogOscModule {
    osc: AnalogOscillator,
    freq: f64,
}

impl AnalogOscModule {
    fn new(sample_rate: f64, freq: f64, mode: u32) -> Self {
        Self {
            osc: AnalogOscillator::new(sample_rate as f32, mode),
            freq,
        }
    }
}

impl Module for AnalogOscModule {
    fn input_count(&self) -> usize { 1 } // freq in
    fn output_count(&self) -> usize { 1 }

    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let block_size = if !inputs.is_empty() && !inputs[0].is_empty() { inputs[0].len() } else { 64 };

        let mut freq_block = vec![self.freq as f32; block_size];
        if !inputs.is_empty() && inputs[0].len() == block_size {
            // we could add modulation here
            for i in 0..block_size {
                if inputs[0][i] != 0.0 {
                    freq_block[i] = inputs[0][i] as f32;
                }
            }
        }

        let mut out = vec![0.0; block_size];
        self.osc.process_block(&freq_block, &mut out);

        vec![out.iter().map(|&x| x as f64).collect()]
    }
}


fn main() {
    let cli = Cli::parse();

    let f = File::open(&cli.patch).expect("Failed to open patch file");
    let patch: Patch = serde_yaml::from_reader(f).expect("Failed to parse patch file");

    println!("Loaded patch: {}", patch.name);

    let sample_rate = patch.sample_rate;
    let block_size = 64;

    let mut graph = ModuleGraph::new(sample_rate as usize, block_size);
    let mut output_filename = String::from("../audio/output.wav");
    let mut output_source_node = String::new();

    // Create modules
    for (id, config) in &patch.modules {
        if config.module_type == "oscillator" {
            let freq = config.parameters.get("frequency").and_then(|v| v.as_f64()).unwrap_or(440.0);
            let waveform = config.parameters.get("waveform").and_then(|v| v.as_str()).unwrap_or("sine");
            let mode = match waveform {
                "noise" => 0,
                "sine" => 1,
                "saw" => 2,
                "triangle" => 3,
                "square" => 4,
                "pulse" => 5,
                _ => 1,
            };
            let osc = Box::new(AnalogOscModule::new(sample_rate, freq, mode));
            graph.add_module(id.clone(), osc);
        } else if config.module_type == "vca" {
            graph.add_module(id.clone(), Box::new(BasicVCA::new()));
        } else if config.module_type == "envelope_adsr" {
            let a = config.parameters.get("attack").and_then(|v| v.as_f64()).unwrap_or(0.01);
            let d = config.parameters.get("decay").and_then(|v| v.as_f64()).unwrap_or(0.1);
            let s = config.parameters.get("sustain").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let r = config.parameters.get("release").and_then(|v| v.as_f64()).unwrap_or(0.1);

            graph.add_module(id.clone(), Box::new(ADSRModule::new(sample_rate, a, d, s, r)));
        } else if config.module_type == "output_wav" {
            if let Some(fname) = config.parameters.get("filename").and_then(|v| v.as_str()) {
                output_filename = format!("../audio/{}", fname);
            }
        } else if config.module_type == "rnbo_oscillator" {
            let mut osc = Box::new(RNBOModule::with_config(sample_rate, block_size));
            let mode = config.parameters.get("mode").and_then(|v| v.as_f64()).unwrap_or(2.0);
            osc.set_parameter(0, mode / 5.0); // Normalize as per Python implementation
            graph.add_module(id.clone(), osc);
        } else {
            // Provide a dummy module so it does not fail
            println!("Warning: unsupported module type {}, skipping implementation", config.module_type);
        }
    }

    // Connections
    for conn in &patch.connections {
        let from_parts: Vec<&str> = conn.from.split('.').collect();
        let to_parts: Vec<&str> = conn.to.split('.').collect();

        let src_id = from_parts[0];
        let src_port = from_parts.get(1).unwrap_or(&"0").parse::<usize>().unwrap_or(0);

        let dst_id = to_parts[0];
        let dst_port = to_parts.get(1).unwrap_or(&"0").parse::<usize>().unwrap_or(0);

        if let Some(config) = patch.modules.get(dst_id) {
            if config.module_type == "output_wav" {
                output_source_node = conn.from.clone();
                continue;
            }
        }

        if graph.nodes.contains_key(src_id) && graph.nodes.contains_key(dst_id) {
            graph.add_connection(src_id, src_port, dst_id, dst_port).unwrap();
        }
    }

    graph.compute_execution_order().unwrap();

    // Determine max length based on sequence
    let mut duration_secs = 2.0;
    for ev in &patch.sequence {
        if ev.time + 1.0 > duration_secs {
            duration_secs = ev.time + 1.0;
        }
    }

    let total_samples = (duration_secs * sample_rate) as usize;

    let mut current_sample = 0;
    let mut seq_idx = 0;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Ensure dir exists
    std::fs::create_dir_all("../audio").unwrap_or_default();

    let mut writer = hound::WavWriter::create(&output_filename, spec).expect("Failed to create wav writer");

    println!("Rendering {} seconds...", duration_secs);

    while current_sample < total_samples {
        let current_time = current_sample as f64 / sample_rate;

        // Handle events
        while seq_idx < patch.sequence.len() && patch.sequence[seq_idx].time <= current_time {
            let ev = &patch.sequence[seq_idx];
            graph.handle_message(&ev.target, &ev.action);
            seq_idx += 1;
        }

        let outputs = graph.process_block();

        // collect output
        let out_node_id = output_source_node.split('.').next().unwrap_or("");
        let out_port = output_source_node.split('.').nth(1).unwrap_or("0").parse::<usize>().unwrap_or(0);

        if let Some(node_outs) = outputs.get(out_node_id) {
            if let Some(block) = node_outs.get(out_port) {
                for &sample in block {
                    let mut s = (sample * std::i16::MAX as f64) as i32;
                    if s > std::i16::MAX as i32 { s = std::i16::MAX as i32; }
                    if s < std::i16::MIN as i32 { s = std::i16::MIN as i32; }
                    writer.write_sample(s as i16).unwrap();
                }
            }
        }

        current_sample += block_size;
    }

    writer.finalize().unwrap();
    println!("Rendered to {}", output_filename);
}
