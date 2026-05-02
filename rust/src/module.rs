use std::collections::HashMap;
use crate::signal_store::SignalStore;

/// Trait defining the behavior of an audio module.
pub trait Module {
    /// Returns the number of input connections this module accepts.
    fn input_count(&self) -> usize;

    /// Returns the number of output connections this module produces.
    fn output_count(&self) -> usize;

    /// Processes a block of audio.
    ///
    /// Inputs are provided as a vector of vectors (each inner vector is a block for one input channel).
    /// Returns a vector of vectors containing the output blocks.
    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>>;

    /// Handle dynamic control messages like trigger/release
    fn handle_message(&mut self, _msg: &str) {}
}

/// A node in the graph wrapping a `Module` with its input and output bus mapping.
pub struct StoreNode {
    pub module_id: String,
    pub module: Box<dyn Module>,
    pub input_keys: HashMap<usize, String>,
    pub output_keys: HashMap<usize, String>,
}

impl StoreNode {
    pub fn new(module_id: String, module: Box<dyn Module>) -> Self {
        Self {
            module_id,
            module,
            input_keys: HashMap::new(),
            output_keys: HashMap::new(),
        }
    }

    pub fn process(&mut self, store: &mut SignalStore, block_size: usize) {
        let input_count = self.module.input_count();
        let mut inputs = Vec::with_capacity(input_count);

        for i in 0..input_count {
            if let Some(key) = self.input_keys.get(&i) {
                inputs.push(store.get_or_zeros(key));
            } else {
                inputs.push(vec![0.0; block_size]);
            }
        }

        let outputs = self.module.process(&inputs);

        for (i, output_signal) in outputs.into_iter().enumerate() {
            if let Some(key) = self.output_keys.get(&i) {
                store.set(key.clone(), output_signal);
            }
        }
    }
}
