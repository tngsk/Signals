import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace("fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {", "fn process(&mut self, inputs: &[&[f64]]) -> Vec<Vec<f64>> {")
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('rust/src/main.rs')
fix_file('rust/src/rnbo_module.rs')
fix_file('rust/benches/benchmark.rs')
fix_file('rust/tests/bench.rs')
fix_file('rust/tests/core_tests.rs')
