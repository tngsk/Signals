"""
StoreNode for adapting core modules to the SignalStore architecture.

This module provides the StoreNode class which wraps an existing Module
and manages its I/O using a SignalStore.
"""


from .module import Module, Signal
from .store import SignalStore


class StoreNode:
    """
    Wrapper for Modules that uses a SignalStore for input/output.

    The StoreNode allows a standard process(inputs) module to be executed
    in a bus-based architecture by specifying which keys in the SignalStore
    correspond to which input/output indices.
    """

    def __init__(
        self,
        module_id: str,
        module: Module,
        input_keys: dict[int, str] | None = None,
        output_keys: dict[int, str] | None = None,
    ):
        """
        Initialize the StoreNode.

        Args:
            module_id: A unique identifier for this node/module.
            module: The core processing Module to wrap.
            input_keys: Mapping of input index to SignalStore key.
            output_keys: Mapping of output index to SignalStore key.
        """
        self.module_id = module_id
        self.module = module
        self.input_keys: dict[int, str] = input_keys or {}
        self.output_keys: dict[int, str] = output_keys or {}

        # Cache for fast processing
        self._inputs_cache: list[Signal] = []

    def process(self, store: SignalStore) -> None:
        """
        Process the module using data from the store.

        Reads inputs from the store based on input_keys, executes the
        wrapped module, and writes outputs to the store based on output_keys.

        Args:
            store: The SignalStore to read from and write to.
        """
        # Gather inputs from store
        inputs = []
        for i in range(self.module.input_count):
            if i in self.input_keys:
                key = self.input_keys[i]
                inputs.append(store.get(key))
            else:
                # Provide a default 0.0 audio signal if not connected
                inputs.append(store.get(f"_unconnected_{self.module_id}_in_{i}"))

        # Execute module
        if inputs or self.module.input_count == 0:
            outputs = self.module.process(inputs if inputs else None)
        else:
            outputs = self.module.process()

        # Write outputs to store
        for i, signal in enumerate(outputs):
            if i in self.output_keys:
                key = self.output_keys[i]
                store.set(key, signal)
