use signals_core::signal_store::SignalStore;
use signals_core::module::Module;
use signals_core::graph::ModuleGraph;
// use signals_core::rnbo_module::RNBOModule;

struct DummyModule;
impl Module for DummyModule {
    fn input_count(&self) -> usize { 1 }
    fn output_count(&self) -> usize { 1 }
    fn process(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let in_sig = &inputs[0];
        let mut out = vec![0.0; in_sig.len()];
        for i in 0..in_sig.len() {
            out[i] = in_sig[i] + 1.0;
        }
        vec![out]
    }
}

#[test]
fn test_signal_store() {
    let mut store = SignalStore::new(4);
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    store.set("test_key".to_string(), signal.clone());
    assert_eq!(store.get("test_key").unwrap(), &signal);
    assert_eq!(store.get_or_zeros("missing_key"), vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_module_graph() {
    let mut graph = ModuleGraph::new(48000, 4);

    graph.add_module("mod1".to_string(), Box::new(DummyModule));
    graph.add_module("mod2".to_string(), Box::new(DummyModule));

    assert!(graph.add_connection("mod1", 0, "mod2", 0).is_ok());
    assert!(graph.compute_execution_order().is_ok());

    assert_eq!(graph.execution_order, vec!["mod1", "mod2"]);

    let outputs = graph.process_block();

    // mod1 has no inputs, gets zeros. Zeros + 1.0 = 1.0
    // mod2 gets 1.0, outputs 2.0
    assert_eq!(outputs.get("mod2").unwrap()[0], vec![2.0, 2.0, 2.0, 2.0]);
}
