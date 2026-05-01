use crate::module::Module;

#[cxx::bridge(namespace = "rnbo_bridge")]
pub mod ffi {
    unsafe extern "C++" {
        include!("signals_core/src/rnbo_bridge.h");

        type RnboHost;

        fn create_rnbo_host(sample_rate: f64, block_size: usize) -> UniquePtr<RnboHost>;

        fn prepare_to_process(self: Pin<&mut RnboHost>, sample_rate: f64, block_size: usize);

        fn set_parameter(self: Pin<&mut RnboHost>, index: usize, value: f64);
        fn get_parameter(self: &RnboHost, index: usize) -> f64;

        fn process_block(self: Pin<&mut RnboHost>, inputs: &[f64], outputs: &mut [f64], block_size: usize);

        fn get_num_inputs(self: &RnboHost) -> usize;
        fn get_num_outputs(self: &RnboHost) -> usize;
    }
}

pub struct RNBOModule {
    inner: cxx::UniquePtr<ffi::RnboHost>,
}

impl RNBOModule {
    pub fn with_config(sample_rate: f64, block_size: usize) -> Self {
        Self {
            inner: ffi::create_rnbo_host(sample_rate, block_size),
        }
    }

    pub fn prepare_to_process(&mut self, sample_rate: f64, block_size: usize) {
        self.inner.pin_mut().prepare_to_process(sample_rate, block_size);
    }

    pub fn set_parameter(&mut self, index: usize, value: f64) {
        self.inner.pin_mut().set_parameter(index, value);
    }

    pub fn get_parameter(&self, index: usize) -> f64 {
        self.inner.get_parameter(index)
    }
}

impl Module for RNBOModule {
    fn input_count(&self) -> usize {
        self.inner.get_num_inputs()
    }

    fn output_count(&self) -> usize {
        self.inner.get_num_outputs()
    }

    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let num_inputs = self.input_count();
        let num_outputs = self.output_count();

        let actual_block_size = if !inputs.is_empty() && !inputs[0].is_empty() {
            inputs[0].len()
        } else {
            64
        };

        let mut flattened_inputs = vec![0.0; num_inputs * actual_block_size];
        for i in 0..num_inputs {
            if i < inputs.len() && inputs[i].len() == actual_block_size {
                let start = i * actual_block_size;
                flattened_inputs[start..start+actual_block_size].copy_from_slice(&inputs[i]);
            }
        }

        let mut flattened_outputs = vec![0.0; num_outputs * actual_block_size];

        self.inner.pin_mut().process_block(&flattened_inputs, &mut flattened_outputs, actual_block_size);

        let mut outputs = Vec::with_capacity(num_outputs);
        for i in 0..num_outputs {
            let start = i * actual_block_size;
            let end = start + actual_block_size;
            outputs.push(flattened_outputs[start..end].to_vec());
        }

        outputs
    }
}
