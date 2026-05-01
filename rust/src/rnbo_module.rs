use crate::module::Module;

#[cxx::bridge(namespace = "rnbo_bridge")]
pub mod ffi {
    unsafe extern "C++" {
        include!("signals_core/src/rnbo_bridge.h");

        type RNBOObject;

        fn create_rnbo_object() -> UniquePtr<RNBOObject>;

        fn set_parameter(self: Pin<&mut RNBOObject>, index: usize, value: f64);
        fn get_parameter(self: &RNBOObject, index: usize) -> f64;

        fn process_block(self: Pin<&mut RNBOObject>, inputs: &[f64], outputs: &mut [f64], block_size: usize);

        fn get_num_inputs(self: &RNBOObject) -> usize;
        fn get_num_outputs(self: &RNBOObject) -> usize;
    }
}

pub struct RNBOModule {
    inner: cxx::UniquePtr<ffi::RNBOObject>,
}

impl RNBOModule {
    pub fn new() -> Self {
        Self {
            inner: ffi::create_rnbo_object(),
        }
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
