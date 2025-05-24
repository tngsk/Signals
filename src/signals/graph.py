"""
Module graph system for managing signal flow and execution order.

This module provides the core graph processing engine that manages module
instantiation, signal routing, and execution scheduling for synthesizer patches.
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, deque
import time

from .module import Module, Signal, SignalType
from .patch import Patch, Connection, SequenceEvent


class GraphError(Exception):
    """Base exception for graph-related errors."""
    pass


class CyclicGraphError(GraphError):
    """Raised when a cyclic dependency is detected in the module graph."""
    pass


class ModuleGraphNode:
    """
    Represents a node in the module graph.
    
    Each node contains a module instance and tracks its connections
    to other nodes in the graph.
    """
    
    def __init__(self, module_id: str, module: Module):
        self.module_id = module_id
        self.module = module
        self.input_connections: Dict[int, Tuple[str, int]] = {}  # input_idx -> (source_id, output_idx)
        self.output_connections: Dict[int, List[Tuple[str, int]]] = defaultdict(list)  # output_idx -> [(dest_id, input_idx)]
        self.cached_outputs: Optional[List[Signal]] = None
        self.processed_this_cycle = False
    
    def add_input_connection(self, input_idx: int, source_id: str, output_idx: int):
        """Add an input connection from another module."""
        self.input_connections[input_idx] = (source_id, output_idx)
    
    def add_output_connection(self, output_idx: int, dest_id: str, input_idx: int):
        """Add an output connection to another module."""
        self.output_connections[output_idx].append((dest_id, input_idx))
    
    def reset_cycle(self):
        """Reset processing state for new processing cycle."""
        self.cached_outputs = None
        self.processed_this_cycle = False


class ModuleGraph:
    """
    Manages the complete module graph and signal processing pipeline.
    
    Handles module instantiation, connection management, topological sorting
    for execution order, and sample-by-sample processing of the entire graph.
    """
    
    def __init__(self, patch: Patch):
        self.patch = patch
        self.nodes: Dict[str, ModuleGraphNode] = {}
        self.execution_order: List[str] = []
        self.sequence_events: List[SequenceEvent] = []
        self.current_time = 0.0
        self.sample_rate = patch.sample_rate
        
        self._build_graph()
        self._compute_execution_order()
        self._prepare_sequence()
    
    def _build_graph(self):
        """Build the module graph from patch definition."""
        # Instantiate all modules
        for module_id, module_data in self.patch.modules.items():
            module_type = module_data['type']
            module_class = self.patch.MODULE_REGISTRY[module_type]
            
            # Create module instance with sample rate
            if module_type == 'output_wav':
                # OutputWav needs filename parameter
                filename = module_data['parameters'].get('filename', f'output_{module_id}.wav')
                module = module_class(filename, self.sample_rate)
            else:
                module = module_class(self.sample_rate)
            
            # Set module parameters
            for param_name, param_value in module_data['parameters'].items():
                if param_name != 'filename':  # Skip filename for output_wav
                    module.set_parameter(param_name, param_value)
            
            # Create graph node
            self.nodes[module_id] = ModuleGraphNode(module_id, module)
        
        # Build connections
        for connection in self.patch.connections:
            self._add_connection(connection)
    
    def _add_connection(self, connection: Connection):
        """Add a connection between two modules in the graph."""
        source_node = self.nodes.get(connection.source_module)
        dest_node = self.nodes.get(connection.dest_module)
        
        if not source_node:
            raise GraphError(f"Source module not found: {connection.source_module}")
        if not dest_node:
            raise GraphError(f"Destination module not found: {connection.dest_module}")
        
        # Convert string indices to integers if needed
        source_output = connection.source_output
        dest_input = connection.dest_input
        
        if isinstance(source_output, str):
            try:
                source_output = int(source_output)
            except ValueError:
                source_output = 0  # Default to first output
        
        if isinstance(dest_input, str):
            try:
                dest_input = int(dest_input)
            except ValueError:
                dest_input = 0  # Default to first input
        
        # Validate connection indices
        if source_output >= source_node.module.output_count:
            raise GraphError(f"Source module {connection.source_module} has no output {source_output}")
        if dest_input >= dest_node.module.input_count:
            raise GraphError(f"Destination module {connection.dest_module} has no input {dest_input}")
        
        # Add bidirectional connection references
        source_node.add_output_connection(source_output, connection.dest_module, dest_input)
        dest_node.add_input_connection(dest_input, connection.source_module, source_output)
    
    def _compute_execution_order(self):
        """Compute topological execution order using Kahn's algorithm."""
        # Calculate in-degrees
        in_degree = {module_id: 0 for module_id in self.nodes.keys()}
        
        for node in self.nodes.values():
            for input_connections in node.input_connections.values():
                source_id = input_connections[0]
                in_degree[node.module_id] += 1
        
        # Find nodes with no incoming edges
        queue = deque([module_id for module_id, degree in in_degree.items() if degree == 0])
        execution_order = []
        
        while queue:
            current_id = queue.popleft()
            execution_order.append(current_id)
            
            # Process all outgoing connections
            current_node = self.nodes[current_id]
            for output_connections in current_node.output_connections.values():
                for dest_id, _ in output_connections:
                    in_degree[dest_id] -= 1
                    if in_degree[dest_id] == 0:
                        queue.append(dest_id)
        
        # Check for cycles
        if len(execution_order) != len(self.nodes):
            remaining_nodes = set(self.nodes.keys()) - set(execution_order)
            raise CyclicGraphError(f"Cyclic dependency detected involving modules: {remaining_nodes}")
        
        self.execution_order = execution_order
    
    def _prepare_sequence(self):
        """Prepare sequence events for processing."""
        self.sequence_events = sorted(self.patch.sequence, key=lambda e: e.time)
    
    def process_sample(self) -> Dict[str, List[Signal]]:
        """
        Process one sample through the entire graph.
        
        Returns dictionary mapping module IDs to their output signals.
        """
        # Reset all nodes for new processing cycle
        for node in self.nodes.values():
            node.reset_cycle()
        
        # Process sequence events for current time
        self._process_sequence_events()
        
        # Process modules in execution order
        outputs = {}
        for module_id in self.execution_order:
            node_outputs = self._process_node(module_id)
            outputs[module_id] = node_outputs
        
        # Advance time
        self.current_time += 1.0 / self.sample_rate
        
        return outputs
    
    def _process_sequence_events(self):
        """Process any sequence events that should occur at current time."""
        current_sample_time = self.current_time
        
        for event in self.sequence_events:
            # Check if event should trigger (within one sample period)
            if abs(event.time - current_sample_time) < (0.5 / self.sample_rate):
                self._execute_sequence_event(event)
    
    def _execute_sequence_event(self, event: SequenceEvent):
        """Execute a single sequence event."""
        target_node = self.nodes.get(event.target)
        if not target_node:
            return  # Skip unknown targets
        
        target_module = target_node.module
        
        if event.action == 'trigger':
            if hasattr(target_module, 'trigger_on'):
                target_module.trigger_on()
        
        elif event.action == 'release':
            if hasattr(target_module, 'trigger_off'):
                target_module.trigger_off()
        
        elif event.action == 'set_parameter':
            for param_name, param_value in event.params.items():
                target_module.set_parameter(param_name, param_value)
    
    def _process_node(self, module_id: str) -> List[Signal]:
        """Process a single node and return its outputs."""
        node = self.nodes[module_id]
        
        # Skip if already processed this cycle
        if node.processed_this_cycle:
            return node.cached_outputs or []
        
        # Gather inputs from connected modules
        inputs = []
        for input_idx in range(node.module.input_count):
            if input_idx in node.input_connections:
                source_id, output_idx = node.input_connections[input_idx]
                source_outputs = self._process_node(source_id)  # Recursive processing
                
                if output_idx < len(source_outputs):
                    inputs.append(source_outputs[output_idx])
                else:
                    # Provide default signal if output doesn't exist
                    inputs.append(Signal(SignalType.AUDIO, 0.0))
            else:
                # Provide default signal for unconnected inputs
                inputs.append(Signal(SignalType.AUDIO, 0.0))
        
        # Process the module
        if inputs or node.module.input_count == 0:
            outputs = node.module.process(inputs if inputs else None)
        else:
            outputs = node.module.process()
        
        # Cache outputs and mark as processed
        node.cached_outputs = outputs
        node.processed_this_cycle = True
        
        return outputs
    
    def process_duration(self, duration: float, 
                        progress_callback: Optional[callable] = None) -> Dict[str, List[List[Signal]]]:
        """
        Process the graph for a specified duration.
        
        Args:
            duration: Duration in seconds to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping module IDs to lists of their output signals over time
        """
        num_samples = int(duration * self.sample_rate)
        all_outputs = defaultdict(list)
        
        for sample_idx in range(num_samples):
            sample_outputs = self.process_sample()
            
            for module_id, outputs in sample_outputs.items():
                all_outputs[module_id].append(outputs)
            
            # Call progress callback if provided
            if progress_callback and sample_idx % 1000 == 0:
                progress = sample_idx / num_samples
                progress_callback(progress)
        
        return dict(all_outputs)
    
    def finalize(self):
        """Finalize all modules that require cleanup."""
        for node in self.nodes.values():
            if hasattr(node.module, 'finalize'):
                node.module.finalize()
    
    def get_module(self, module_id: str) -> Optional[Module]:
        """Get module instance by ID."""
        node = self.nodes.get(module_id)
        return node.module if node else None
    
    def set_module_parameter(self, module_id: str, param_name: str, value: Any):
        """Set parameter for a specific module."""
        module = self.get_module(module_id)
        if module:
            module.set_parameter(param_name, value)
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure."""
        return {
            'module_count': len(self.nodes),
            'connection_count': len(self.patch.connections),
            'execution_order': self.execution_order,
            'sample_rate': self.sample_rate,
            'sequence_events': len(self.sequence_events)
        }