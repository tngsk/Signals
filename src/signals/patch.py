"""
Patch system for loading and managing synthesizer configurations.

This module provides functionality for loading YAML patch files with Jinja2
template support, validating patch configurations, and managing module
instantiation from patch definitions.
"""

import yaml
from jinja2 import Template, Environment, meta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import re

from .module import Module, Signal, SignalType
from .oscillator import Oscillator, WaveformType
from .envelope import EnvelopeADSR
from .mixer import Mixer
from .output import OutputWav


class PatchError(Exception):
    """Base exception for patch-related errors."""
    pass


class PatchValidationError(PatchError):
    """Raised when patch validation fails."""
    pass


class PatchTemplateError(PatchError):
    """Raised when template processing fails."""
    pass


class Connection:
    """
    Represents a connection between two modules in a patch.
    
    Attributes:
        source_module: ID of the source module
        source_output: Output index or name (default: 0)
        dest_module: ID of the destination module  
        dest_input: Input index or name (default: 0)
    """
    
    def __init__(self, source_module: str, dest_module: str,
                 source_output: Union[int, str] = 0,
                 dest_input: Union[int, str] = 0):
        self.source_module = source_module
        self.source_output = source_output
        self.dest_module = dest_module
        self.dest_input = dest_input
    
    def __repr__(self):
        return f"Connection({self.source_module}.{self.source_output} -> {self.dest_module}.{self.dest_input})"


class SequenceEvent:
    """
    Represents a timed event in a patch sequence.
    
    Attributes:
        time: Event time in seconds
        action: Action type ('trigger', 'release', 'set_parameter')
        target: Target module ID
        params: Additional parameters for the action
    """
    
    def __init__(self, time: float, action: str, target: str, params: Optional[Dict] = None):
        self.time = time
        self.action = action
        self.target = target
        self.params = params or {}
    
    def __repr__(self):
        return f"SequenceEvent({self.time}s: {self.action} -> {self.target})"


class PatchTemplate:
    """
    Handles YAML patch files with Jinja2 template variables.
    
    Supports parameterized patches where variables can be substituted
    at load time to create variations of the same base patch.
    """
    
    def __init__(self, template_file: Union[str, Path]):
        self.template_file = Path(template_file)
        self.template_content = self._load_template()
        self.variables = self._extract_variables()
    
    def _load_template(self) -> str:
        """Load template content from file."""
        try:
            return self.template_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise PatchTemplateError(f"Template file not found: {self.template_file}")
        except Exception as e:
            raise PatchTemplateError(f"Error reading template file: {e}")
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from Jinja2 template."""
        env = Environment()
        try:
            ast = env.parse(self.template_content)
            return list(meta.find_undeclared_variables(ast))
        except Exception as e:
            raise PatchTemplateError(f"Error parsing template: {e}")
    
    def get_variable_schema(self) -> Dict[str, Any]:
        """
        Get schema information for template variables.
        
        Returns dictionary with variable names and their default values
        if specified in the template's variables section.
        """
        # Try to parse a basic version to extract variables section
        try:
            basic_content = Template(self.template_content).render()
            basic_patch = yaml.safe_load(basic_content)
            return basic_patch.get('variables', {})
        except:
            # If template rendering fails, return discovered variables
            return {var: None for var in self.variables}
    
    def instantiate(self, variables: Optional[Dict[str, Any]] = None) -> 'Patch':
        """
        Instantiate patch from template with given variables.
        
        Args:
            variables: Dictionary of variable values to substitute
            
        Returns:
            Patch instance with template variables substituted
        """
        variables = variables or {}
        
        try:
            template = Template(self.template_content)
            rendered_content = template.render(**variables)
            patch_data = yaml.safe_load(rendered_content)
            return Patch.from_dict(patch_data, source_file=self.template_file)
        except yaml.YAMLError as e:
            raise PatchTemplateError(f"YAML parsing error: {e}")
        except Exception as e:
            raise PatchTemplateError(f"Template rendering error: {e}")


class Patch:
    """
    Represents a complete synthesizer patch configuration.
    
    A patch defines modules, their parameters, connections between modules,
    and optional timed sequences for automation.
    """
    
    # Registry of available module types
    MODULE_REGISTRY = {
        'oscillator': Oscillator,
        'envelope_adsr': EnvelopeADSR,
        'mixer': Mixer,
        'output_wav': OutputWav,
    }
    
    def __init__(self, name: str = "Untitled Patch", description: str = "",
                 sample_rate: int = 48000):
        self.name = name
        self.description = description
        self.sample_rate = sample_rate
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.connections: List[Connection] = []
        self.sequence: List[SequenceEvent] = []
        self.source_file: Optional[Path] = None
    
    @classmethod
    def from_file(cls, patch_file: Union[str, Path]) -> 'Patch':
        """Load patch from YAML file."""
        patch_file = Path(patch_file)
        
        try:
            content = patch_file.read_text(encoding='utf-8')
            patch_data = yaml.safe_load(content)
            return cls.from_dict(patch_data, source_file=patch_file)
        except FileNotFoundError:
            raise PatchError(f"Patch file not found: {patch_file}")
        except yaml.YAMLError as e:
            raise PatchError(f"YAML parsing error in {patch_file}: {e}")
        except Exception as e:
            raise PatchError(f"Error loading patch file {patch_file}: {e}")
    
    @classmethod
    def from_dict(cls, patch_data: Dict[str, Any], 
                  source_file: Optional[Path] = None) -> 'Patch':
        """Create patch from dictionary data."""
        patch = cls()
        patch.source_file = source_file
        
        # Basic metadata
        patch.name = patch_data.get('name', 'Untitled Patch')
        patch.description = patch_data.get('description', '')
        patch.sample_rate = patch_data.get('sample_rate', 48000)
        
        # Load modules
        modules_data = patch_data.get('modules', {})
        for module_id, module_config in modules_data.items():
            patch._add_module(module_id, module_config)
        
        # Load connections
        connections_data = patch_data.get('connections', [])
        for conn_data in connections_data:
            patch._add_connection(conn_data)
        
        # Load sequence
        sequence_data = patch_data.get('sequence', [])
        for event_data in sequence_data:
            patch._add_sequence_event(event_data)
        
        # Validate the complete patch
        patch.validate()
        
        return patch
    
    def _add_module(self, module_id: str, module_config: Dict[str, Any]):
        """Add module definition to patch."""
        module_type = module_config.get('type')
        if not module_type:
            raise PatchValidationError(f"Module {module_id} missing 'type' field")
        
        if module_type not in self.MODULE_REGISTRY:
            raise PatchValidationError(
                f"Unknown module type '{module_type}' for module {module_id}. "
                f"Available types: {list(self.MODULE_REGISTRY.keys())}"
            )
        
        self.modules[module_id] = {
            'type': module_type,
            'parameters': module_config.get('parameters', {}),
            'config': module_config
        }
    
    def _add_connection(self, conn_data: Dict[str, Any]):
        """Add connection definition to patch."""
        if 'from' not in conn_data or 'to' not in conn_data:
            raise PatchValidationError("Connection missing 'from' or 'to' field")
        
        # Parse source and destination
        source_parts = conn_data['from'].split('.')
        dest_parts = conn_data['to'].split('.')
        
        if len(source_parts) < 1 or len(dest_parts) < 1:
            raise PatchValidationError("Invalid connection format. Use 'module.output' -> 'module.input'")
        
        source_module = source_parts[0]
        source_output = source_parts[1] if len(source_parts) > 1 else 0
        dest_module = dest_parts[0]
        dest_input = dest_parts[1] if len(dest_parts) > 1 else 0
        
        # Convert to int if numeric
        try:
            source_output = int(source_output)
        except ValueError:
            pass
        
        try:
            dest_input = int(dest_input)
        except ValueError:
            pass
        
        connection = Connection(source_module, dest_module, source_output, dest_input)
        self.connections.append(connection)
    
    def _add_sequence_event(self, event_data: Dict[str, Any]):
        """Add sequence event to patch."""
        required_fields = ['time', 'action', 'target']
        for field in required_fields:
            if field not in event_data:
                raise PatchValidationError(f"Sequence event missing '{field}' field")
        
        event = SequenceEvent(
            time=float(event_data['time']),
            action=event_data['action'],
            target=event_data['target'],
            params=event_data.get('params', {})
        )
        self.sequence.append(event)
    
    def validate(self):
        """Validate patch configuration."""
        # Check that all modules referenced in connections exist
        all_module_ids = set(self.modules.keys())
        
        for conn in self.connections:
            if conn.source_module not in all_module_ids:
                raise PatchValidationError(f"Connection references unknown source module: {conn.source_module}")
            if conn.dest_module not in all_module_ids:
                raise PatchValidationError(f"Connection references unknown destination module: {conn.dest_module}")
        
        # Check that all modules referenced in sequence exist
        for event in self.sequence:
            if event.target not in all_module_ids:
                raise PatchValidationError(f"Sequence event references unknown module: {event.target}")
        
        # Sort sequence by time
        self.sequence.sort(key=lambda e: e.time)
    
    def get_module_count(self) -> int:
        """Get total number of modules in patch."""
        return len(self.modules)
    
    def get_connection_count(self) -> int:
        """Get total number of connections in patch."""
        return len(self.connections)
    
    def get_duration(self) -> float:
        """Get total duration based on sequence events."""
        if not self.sequence:
            return 0.0
        return max(event.time for event in self.sequence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert patch back to dictionary format."""
        result = {
            'name': self.name,
            'description': self.description,
            'sample_rate': self.sample_rate,
            'modules': {}
        }
        
        # Convert modules
        for module_id, module_data in self.modules.items():
            result['modules'][module_id] = {
                'type': module_data['type'],
                'parameters': module_data['parameters']
            }
        
        # Convert connections
        if self.connections:
            result['connections'] = []
            for conn in self.connections:
                result['connections'].append({
                    'from': f"{conn.source_module}.{conn.source_output}",
                    'to': f"{conn.dest_module}.{conn.dest_input}"
                })
        
        # Convert sequence
        if self.sequence:
            result['sequence'] = []
            for event in self.sequence:
                event_dict = {
                    'time': event.time,
                    'action': event.action,
                    'target': event.target
                }
                if event.params:
                    event_dict['params'] = event.params
                result['sequence'].append(event_dict)
        
        return result
    
    def __repr__(self):
        return f"Patch('{self.name}', {self.get_module_count()} modules, {self.get_connection_count()} connections)"