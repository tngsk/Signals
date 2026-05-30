import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    content = content.replace("node_outputs.push(self.store.get_or_zeros(bus_key).to_vec());", "node_outputs.push(self.store.get_or_zeros(bus_key).to_vec());")
    content = content.replace("node_outputs.push(self.store.get_zeros().to_vec());", "node_outputs.push(self.store.get_zeros().to_vec());")
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('rust/src/graph.rs')
