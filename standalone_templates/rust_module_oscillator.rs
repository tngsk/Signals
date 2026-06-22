/// オーディオモジュールが実装すべき基本トレイト
pub trait Module {
    /// 受け取る入力の数を返します
    fn input_count(&self) -> usize;

    /// 生成する出力の数を返します
    fn output_count(&self) -> usize;

    /// オーディオブロックを処理します
    /// inputs: 各入力チャンネルのオーディオブロック（Vec<f64>）の配列
    /// 戻り値: 各出力チャンネルのオーディオブロック（Vec<f64>）の配列
    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>>;
}

/// シンプルなサイン波オシレーター
pub struct SineOscillator {
    sample_rate: f64,
    frequency: f64,
    phase: f64,
    block_size: usize,
}

impl SineOscillator {
    pub fn new(sample_rate: f64, frequency: f64, block_size: usize) -> Self {
        Self {
            sample_rate,
            frequency,
            phase: 0.0,
            block_size,
        }
    }
}

impl Module for SineOscillator {
    // オシレーターは入力を受け取らない（CV入力を考えない最もシンプルな場合）
    fn input_count(&self) -> usize {
        0
    }

    // 1つのオーディオ出力（モノラル）を生成する
    fn output_count(&self) -> usize {
        1
    }

    fn process(&mut self, _inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut output = vec![0.0; self.block_size];
        let phase_increment = 2.0 * std::f64::consts::PI * self.frequency / self.sample_rate;

        for i in 0..self.block_size {
            output[i] = self.phase.sin();
            self.phase += phase_increment;

            // 位相が2πを超えないようにラップアラウンド
            if self.phase >= 2.0 * std::f64::consts::PI {
                self.phase -= 2.0 * std::f64::consts::PI;
            }
        }

        // 1チャンネルの出力として返す
        vec![output]
    }
}

// --- 使用例 ---
fn main() {
    let sample_rate = 44100.0;
    let frequency = 440.0; // A4 (440Hz)
    let block_size = 64;

    let mut osc = SineOscillator::new(sample_rate, frequency, block_size);

    // 最初のオーディオブロックを生成
    let outputs = osc.process(&[]); // 入力なし

    println!("Generated {} output channels.", outputs.len());
    println!("Channel 0 length: {} samples.", outputs[0].len());

    // 最初の数サンプルを表示
    for i in 0..5 {
        println!("Sample {}: {:.4}", i, outputs[0][i]);
    }
}
