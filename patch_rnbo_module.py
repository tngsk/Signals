with open("rust/src/rnbo_module.rs", "r") as f:
    content = f.read()

content = content.replace('fn create_rnbo_host(sample_rate: f64, block_size: usize) -> UniquePtr<RnboHost>;', 'fn create_rnbo_host(sample_rate: f64, block_size: usize) -> UniquePtr<RnboHost>;\n\n        fn prepare_to_process(self: Pin<&mut RnboHost>, sample_rate: f64, block_size: usize);')
content = content.replace('pub fn set_parameter(&mut self, index: usize, value: f64) {', 'pub fn prepare_to_process(&mut self, sample_rate: f64, block_size: usize) {\n        self.inner.pin_mut().prepare_to_process(sample_rate, block_size);\n    }\n\n    pub fn set_parameter(&mut self, index: usize, value: f64) {')

with open("rust/src/rnbo_module.rs", "w") as f:
    f.write(content)
