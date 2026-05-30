use signals_core::graph::ModuleGraph;
use signals_core::module::Module;
use std::time::Instant;

struct DummyModule;
impl Module for DummyModule {
    fn input_count(&self) -> usize { 1 }
    fn output_count(&self) -> usize { 1 }
    fn process(&mut self, inputs: &[&[f64]]) -> Vec<Vec<f64>> {
        let in_sig = &inputs[0];
        let mut out = vec![0.0; in_sig.len()];
        for i in 0..in_sig.len() {
            out[i] = in_sig[i] + 1.0;
        }
        vec![out]
    }
}

#[test]
fn bench_process_block() {
    let mut graph = ModuleGraph::new(48000, 64);
    for i in 0..100 {
        graph.add_module(format!("mod{}", i), Box::new(DummyModule));
        if i > 0 {
            graph.add_connection(&format!("mod{}", i-1), 0, &format!("mod{}", i), 0).unwrap();
        }
    }
    graph.compute_execution_order().unwrap();

    let start = Instant::now();
    for _ in 0..10000 {
        graph.process_block();
    }
    println!("Elapsed: {:?}", start.elapsed());
}
