1. **Core Layer: Add SignalStore class**
   - Create `src/signals/core/store.py`
   - Implement `SignalStore` class (which manages a shared dictionary of signals indexed by keys/buses).

2. **Wrap Existing Modules**
   - In `src/signals/core/module.py` (or a new file like `src/signals/core/store_node.py`), implement `StoreNode` class.
   - `StoreNode` will wrap a `Module` instance.
   - It will have `input_keys` (mapping module input index to a bus key) and `output_keys` (mapping module output index to a bus key).
   - Its `process(store: SignalStore)` method will read inputs from `store` via `input_keys`, call the inner `Module.process(inputs)`, and then write the results to `store` via `output_keys`.

3. **Re-implement Graph class**
   - Modify `src/signals/processing/graph.py`.
   - Update `ModuleGraphNode` to use `StoreNode` concept.
   - Change `_build_graph` to instantiate `StoreNode`s. For `connections` (`from: osc1.0`, `to: filter1.0`), generate shared bus keys (e.g. `bus_osc1_0`) and assign them to the corresponding `output_keys` of the source node and `input_keys` of the destination node.
   - Change `process_sample` to use the new flat loop approach: loop over `execution_order`, call `node.process(self.store)`, and return outputs collected from the store.
   - Remove `_process_node` and `_process_node_optimized` recursive logic.
   - Maintain backward compatibility with the current YAML patch format so tests continue to pass.

4. **CLI Entry Points**
   - Add new entry point functions in `src/signals/cli.py` for standard modules (e.g., `oscillator_cli`, `filter_cli`).
   - Use `argparse` to create thin wrappers that set up a simple `WavReadNode -> Module -> WavWriteNode` pipeline using `SignalStore`. (Since WavReadNode might not exist, I may need to create a simple input node or just read the file and put it in the store).
   - Register them in `pyproject.toml` under `[project.scripts]` (e.g., `signals-osc`, `signals-filter`).

5. **Pre-commit and tests**
   - Ensure all tests pass (`uv run pytest tests/`).
   - Run pre-commit checks (`uv run ruff check --fix`).
   - Verify the functionality manually.
