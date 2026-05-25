use std::collections::HashMap;

/// シンプルなブロックベースのシグナルストア（オーディオ信号の共有バス）
///
/// モジュール間でオーディオ信号（f64の配列）を受け渡しするために使用します。
/// 各モジュールはここから入力を読み取り、処理結果をここに書き込みます。
pub struct SignalStore {
    signals: HashMap<String, Vec<f64>>,
    block_size: usize,
}

impl SignalStore {
    /// 新しいSignalStoreを作成します
    /// block_size: 一度に処理するオーディオサンプルの数（例: 64, 128, 256）
    pub fn new(block_size: usize) -> Self {
        Self {
            signals: HashMap::new(),
            block_size,
        }
    }

    /// 指定したキーのシグナルを取得します
    pub fn get(&self, key: &str) -> Option<&Vec<f64>> {
        self.signals.get(key)
    }

    /// 指定したキーのシグナルを取得します。
    /// まだシグナルが存在しない場合は、無音（すべて0.0）のブロックを返します。
    pub fn get_or_zeros(&self, key: &str) -> Vec<f64> {
        self.signals.get(key).cloned().unwrap_or_else(|| vec![0.0; self.block_size])
    }

    /// シグナルをストアに書き込みます。
    /// 注意: シグナルの長さは block_size と一致している必要があります。
    pub fn set(&mut self, key: String, signal: Vec<f64>) {
        if signal.len() != self.block_size {
            panic!("Signal block size mismatch. Expected {}, got {}", self.block_size, signal.len());
        }
        self.signals.insert(key, signal);
    }

    /// ストア内のすべてのシグナルをクリアします。
    /// 通常、次のブロック処理の前に呼び出します。
    pub fn clear(&mut self) {
        self.signals.clear();
    }
}

// --- 使用例 ---
fn main() {
    let block_size = 64;
    let mut store = SignalStore::new(block_size);

    // モジュールAが生成した信号を保存
    let dummy_signal = vec![0.5; block_size]; // 0.5のDCオフセット信号
    store.set("moduleA_out".to_string(), dummy_signal);

    // モジュールBがモジュールAの出力を読み取る
    let input_for_b = store.get_or_zeros("moduleA_out");
    println!("Read {} samples from moduleA_out.", input_for_b.len());
    println!("First sample: {}", input_for_b[0]);

    // 存在しないキーを読み取ると0.0で埋められた配列が返る
    let empty_input = store.get_or_zeros("non_existent");
    println!("Read non_existent: first sample is {}", empty_input[0]);
}
