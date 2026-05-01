use std::collections::{HashMap, HashSet, VecDeque};
use crate::module::{StoreNode, Module};
use crate::signal_store::SignalStore;

pub struct ModuleGraph {
    pub nodes: HashMap<String, StoreNode>,
    pub store: SignalStore,
    pub execution_order: Vec<String>,
    pub sample_rate: usize,
    pub block_size: usize,
    pub input_connections: HashMap<String, HashMap<usize, (String, usize)>>,
    pub output_connections: HashMap<String, HashMap<usize, Vec<(String, usize)>>>,
}

impl ModuleGraph {
    pub fn new(sample_rate: usize, block_size: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            store: SignalStore::new(block_size),
            execution_order: Vec::new(),
            sample_rate,
            block_size,
            input_connections: HashMap::new(),
            output_connections: HashMap::new(),
        }
    }

    pub fn add_module(&mut self, module_id: String, module: Box<dyn Module>) {
        let mut node = StoreNode::new(module_id.clone(), module);

        for i in 0..node.module.output_count() {
            let bus_key = format!("wire_{}_{}", module_id, i);
            node.output_keys.insert(i, bus_key);
        }

        self.nodes.insert(module_id.clone(), node);
        self.input_connections.insert(module_id.clone(), HashMap::new());
        self.output_connections.insert(module_id.clone(), HashMap::new());
    }

    pub fn add_connection(&mut self, source_id: &str, source_output: usize, dest_id: &str, dest_input: usize) -> Result<(), String> {
        if !self.nodes.contains_key(source_id) {
            return Err(format!("Source module not found: {}", source_id));
        }
        if !self.nodes.contains_key(dest_id) {
            return Err(format!("Destination module not found: {}", dest_id));
        }

        if source_output >= self.nodes[source_id].module.output_count() {
            return Err(format!("Source module {} has no output {}", source_id, source_output));
        }
        if dest_input >= self.nodes[dest_id].module.input_count() {
            return Err(format!("Destination module {} has no input {}", dest_id, dest_input));
        }

        self.output_connections.get_mut(source_id).unwrap().entry(source_output).or_default().push((dest_id.to_string(), dest_input));
        self.input_connections.get_mut(dest_id).unwrap().insert(dest_input, (source_id.to_string(), source_output));

        let bus_key = format!("wire_{}_{}", source_id, source_output);
        self.nodes.get_mut(dest_id).unwrap().input_keys.insert(dest_input, bus_key);

        Ok(())
    }

    pub fn compute_execution_order(&mut self) -> Result<(), String> {
        let mut in_degree = HashMap::new();
        for id in self.nodes.keys() {
            in_degree.insert(id.clone(), 0);
        }

        for (node_id, inputs) in &self.input_connections {
            if let Some(degree) = in_degree.get_mut(node_id) {
                *degree += inputs.len();
            }
        }

        let mut queue: VecDeque<String> = VecDeque::new();
        for (id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(id.clone());
            }
        }

        let mut execution_order = Vec::new();

        while let Some(current_id) = queue.pop_front() {
            execution_order.push(current_id.clone());

            if let Some(outputs) = self.output_connections.get(&current_id) {
                for dests in outputs.values() {
                    for (dest_id, _) in dests {
                        if let Some(degree) = in_degree.get_mut(dest_id) {
                            *degree -= 1;
                            if *degree == 0 {
                                queue.push_back(dest_id.clone());
                            }
                        }
                    }
                }
            }
        }

        if execution_order.len() != self.nodes.len() {
            let all_nodes: HashSet<_> = self.nodes.keys().cloned().collect();
            let ordered_nodes: HashSet<_> = execution_order.iter().cloned().collect();
            let remaining: Vec<_> = all_nodes.difference(&ordered_nodes).cloned().collect();
            return Err(format!("Cyclic dependency detected involving modules: {:?}", remaining));
        }

        self.execution_order = execution_order;
        Ok(())
    }

    pub fn process_block(&mut self) -> HashMap<String, Vec<Vec<f64>>> {
        let mut outputs = HashMap::new();
        let order = self.execution_order.clone();

        for module_id in order {
            if let Some(node) = self.nodes.get_mut(&module_id) {
                node.process(&mut self.store, self.block_size);

                let mut node_outputs = Vec::new();
                for i in 0..node.module.output_count() {
                    if let Some(bus_key) = node.output_keys.get(&i) {
                        node_outputs.push(self.store.get_or_zeros(bus_key));
                    } else {
                        node_outputs.push(vec![0.0; self.block_size]);
                    }
                }
                outputs.insert(module_id.clone(), node_outputs);
            }
        }

        outputs
    }
}
