use std::collections::HashMap;

/// Block-based signal store for managing shared data buses.
pub struct SignalStore {
    signals: HashMap<String, Vec<f64>>,
    block_size: usize,
    zero_buffer: Vec<f64>,
}

impl SignalStore {
    pub fn new(block_size: usize) -> Self {
        Self {
            signals: HashMap::new(),
            block_size,
            zero_buffer: vec![0.0; block_size],
        }
    }

    pub fn get(&self, key: &str) -> Option<&Vec<f64>> {
        self.signals.get(key)
    }

    pub fn get_or_zeros(&self, key: &str) -> &[f64] {
        self.signals.get(key).map(|v| v.as_slice()).unwrap_or(&self.zero_buffer)
    }

    pub fn get_zeros(&self) -> &[f64] {
        &self.zero_buffer
    }

    pub fn set(&mut self, key: String, signal: Vec<f64>) {
        if signal.len() != self.block_size {
            panic!("Signal block size mismatch. Expected {}, got {}", self.block_size, signal.len());
        }
        self.signals.insert(key, signal);
    }

    pub fn clear(&mut self) {
        self.signals.clear();
    }
}
