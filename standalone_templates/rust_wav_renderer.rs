// 注意: このコードを実行するには `hound` クレートが必要です。
// プロジェクトに組み込む場合は Cargo.toml に `hound = "3.4"` などを追加してください。

/*
[dependencies]
hound = "3.4"
*/

/// オーディオバッファ（-1.0 ~ 1.0のf64配列）をWAVファイルとして保存します。
pub fn save_to_wav(filename: &str, sample_rate: u32, samples: &[f64]) -> Result<(), String> {
    // houndクレートを使用してWAVのフォーマットを指定します
    // (ここでは利用できないためコメントアウトしていますが、実際の使用時のテンプレートです)
    /*
    let spec = hound::WavSpec {
        channels: 1, // モノラル
        sample_rate: sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(filename, spec)
        .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

    for &sample in samples {
        // f64 (-1.0 ~ 1.0) を 16bit 整数 (i16) に変換
        // i16の範囲は -32768 ~ 32767
        let mut s = (sample * std::i16::MAX as f64) as i32;

        // クリッピング処理（範囲外の値を制限）
        if s > std::i16::MAX as i32 { s = std::i16::MAX as i32; }
        if s < std::i16::MIN as i32 { s = std::i16::MIN as i32; }

        writer.write_sample(s as i16)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer.finalize().map_err(|e| format!("Failed to finalize WAV file: {}", e))?;
    */

    // シミュレーションとしての出力
    println!("Saving {} samples to {} at {}Hz (Simulation)", samples.len(), filename, sample_rate);
    println!("(Uncomment hound code to actualy generate WAV file)");

    Ok(())
}

// --- 使用例 ---
fn main() {
    let sample_rate = 44100;
    let duration_secs = 1.0;
    let frequency = 440.0;

    let total_samples = (sample_rate as f64 * duration_secs) as usize;
    let mut buffer = Vec::with_capacity(total_samples);

    // 1秒間のサイン波を生成
    let phase_increment = 2.0 * std::f64::consts::PI * frequency / sample_rate as f64;
    let mut phase: f64 = 0.0;

    for _ in 0..total_samples {
        buffer.push(phase.sin());
        phase += phase_increment;
        if phase >= 2.0 * std::f64::consts::PI {
            phase -= 2.0 * std::f64::consts::PI;
        }
    }

    // WAVファイルとして出力
    if let Err(e) = save_to_wav("output_template.wav", sample_rate, &buffer) {
        eprintln!("Error saving WAV: {}", e);
    }
}
