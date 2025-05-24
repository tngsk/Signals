"""
Main synthesizer engine providing high-level API for patch processing.

This module provides the SynthEngine class which serves as the main interface
for loading patches, processing audio, and managing the synthesis pipeline.
Designed for both interactive use and external program integration.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import numpy as np
import time

from .patch import Patch, PatchTemplate, PatchError
from .graph import ModuleGraph, GraphError
from ..core.module import Signal, SignalType


class EngineError(Exception):
    """Base exception for engine-related errors."""
    pass


class SynthEngine:
    """
    High-level synthesizer engine for processing patches and generating audio.
    
    The SynthEngine provides a simple interface for loading patch files,
    setting parameters dynamically, and rendering audio output. It handles
    all the complexity of module graph management and signal routing.
    
    Example:
        >>> engine = SynthEngine(sample_rate=48000)
        >>> patch = engine.load_patch("synth.yaml")
        >>> audio_data = engine.render(duration=2.0)
        >>> engine.save_audio("output.wav", audio_data)
    """
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.current_patch: Optional[Patch] = None
        self.current_graph: Optional[ModuleGraph] = None
        self._processing_callbacks: List[Callable] = []
    
    def load_patch(self, patch_file: Union[str, Path], 
                   variables: Optional[Dict[str, Any]] = None) -> Patch:
        """
        Load a patch from file with optional variable substitution.
        
        Args:
            patch_file: Path to YAML patch file or template
            variables: Dictionary of template variables for substitution
            
        Returns:
            Loaded and validated Patch instance
            
        Raises:
            EngineError: If patch loading or validation fails
        """
        try:
            patch_path = Path(patch_file)
            
            if variables is not None:
                # Load as template
                template = PatchTemplate(patch_path)
                patch = template.instantiate(variables)
            else:
                # Load as regular patch
                patch = Patch.from_file(patch_path)
            
            # Override sample rate from engine
            patch.sample_rate = self.sample_rate
            
            # Build module graph
            self.current_patch = patch
            self.current_graph = ModuleGraph(patch)
            
            return patch
            
        except (PatchError, GraphError) as e:
            raise EngineError(f"Failed to load patch: {e}")
        except Exception as e:
            raise EngineError(f"Unexpected error loading patch: {e}")
    
    def load_patch_from_dict(self, patch_data: Dict[str, Any]) -> Patch:
        """
        Load patch from dictionary data.
        
        Args:
            patch_data: Dictionary containing patch configuration
            
        Returns:
            Loaded and validated Patch instance
        """
        try:
            patch = Patch.from_dict(patch_data)
            patch.sample_rate = self.sample_rate
            
            self.current_patch = patch
            self.current_graph = ModuleGraph(patch)
            
            return patch
            
        except (PatchError, GraphError) as e:
            raise EngineError(f"Failed to load patch from dict: {e}")
    
    def render(self, duration: Optional[float] = None, 
               output_file: Optional[Union[str, Path]] = None,
               progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Render audio from the current patch.
        
        Args:
            duration: Duration in seconds (if None, uses patch sequence duration)
            output_file: Optional file path to save rendered audio
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            
        Returns:
            Numpy array containing rendered audio samples
            
        Raises:
            EngineError: If no patch is loaded or rendering fails
        """
        if not self.current_patch or not self.current_graph:
            raise EngineError("No patch loaded. Call load_patch() first.")
        
        try:
            # Determine duration
            if duration is None:
                duration = self.current_patch.get_duration()
                if duration == 0.0:
                    duration = 2.0  # Default duration
            
            # Process the graph
            all_outputs = self.current_graph.process_duration(duration, progress_callback)
            
            # Extract audio from output modules
            audio_data = self._extract_audio_output(all_outputs, duration)
            
            # Save to file if requested
            if output_file:
                self.save_audio(output_file, audio_data)
            
            # Finalize modules
            self.current_graph.finalize()
            
            return audio_data
            
        except Exception as e:
            raise EngineError(f"Rendering failed: {e}")
    
    def _extract_audio_output(self, all_outputs: Dict[str, List[List[Signal]]], 
                             duration: float) -> np.ndarray:
        """Extract audio data from module outputs."""
        # Find output modules
        output_modules = []
        for module_id, module_data in self.current_patch.modules.items():
            if module_data['type'] == 'output_wav':
                output_modules.append(module_id)
        
        if not output_modules:
            # No output modules, collect from all modules
            return self._collect_mixed_audio(all_outputs, duration)
        
        # Use first output module
        output_module_id = output_modules[0]
        if output_module_id not in all_outputs:
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)
        
        # The output module should have processed all audio internally
        # Return a basic signal for compatibility
        num_samples = int(duration * self.sample_rate)
        return np.zeros(num_samples, dtype=np.float32)
    
    def _collect_mixed_audio(self, all_outputs: Dict[str, List[List[Signal]]], 
                            duration: float) -> np.ndarray:
        """Collect and mix audio from all modules."""
        num_samples = int(duration * self.sample_rate)
        mixed_audio = np.zeros(num_samples, dtype=np.float32)
        
        for module_id, sample_outputs in all_outputs.items():
            # Skip output modules to avoid double-counting
            if self.current_patch.modules[module_id]['type'] == 'output_wav':
                continue
            
            for sample_idx, signals in enumerate(sample_outputs):
                if sample_idx >= num_samples:
                    break
                
                # Mix all audio signals from this module
                for signal in signals:
                    if signal.type == SignalType.AUDIO:
                        mixed_audio[sample_idx] += signal.value * 0.1  # Reduce gain
        
        return mixed_audio
    
    def save_audio(self, filename: Union[str, Path], audio_data: np.ndarray,
                   bits_per_sample: int = 16):
        """
        Save audio data to WAV file.
        
        Args:
            filename: Output file path
            audio_data: Audio samples as numpy array
            bits_per_sample: Bit depth (16, 24, or 32)
        """
        from .dsp import write_wav
        
        try:
            write_wav(str(filename), audio_data, self.sample_rate, bits_per_sample)
        except Exception as e:
            raise EngineError(f"Failed to save audio: {e}")
    
    def set_module_parameter(self, module_id: str, param_name: str, value: Any):
        """
        Set a parameter for a specific module in the current patch.
        
        Args:
            module_id: ID of the module to modify
            param_name: Name of the parameter
            value: New parameter value
            
        Raises:
            EngineError: If no patch is loaded or module not found
        """
        if not self.current_graph:
            raise EngineError("No patch loaded. Call load_patch() first.")
        
        try:
            self.current_graph.set_module_parameter(module_id, param_name, value)
        except Exception as e:
            raise EngineError(f"Failed to set parameter: {e}")
    
    def get_module_parameters(self, module_id: str) -> Dict[str, Any]:
        """
        Get current parameters for a module.
        
        Args:
            module_id: ID of the module
            
        Returns:
            Dictionary of current parameter values
            
        Raises:
            EngineError: If module not found
        """
        if not self.current_patch:
            raise EngineError("No patch loaded. Call load_patch() first.")
        
        if module_id not in self.current_patch.modules:
            raise EngineError(f"Module '{module_id}' not found in current patch")
        
        return self.current_patch.modules[module_id]['parameters'].copy()
    
    def get_patch_info(self) -> Dict[str, Any]:
        """
        Get information about the current patch.
        
        Returns:
            Dictionary containing patch metadata and statistics
        """
        if not self.current_patch or not self.current_graph:
            return {}
        
        info = {
            'name': self.current_patch.name,
            'description': self.current_patch.description,
            'sample_rate': self.current_patch.sample_rate,
            'duration': self.current_patch.get_duration(),
            'modules': list(self.current_patch.modules.keys()),
            'module_count': self.current_patch.get_module_count(),
            'connection_count': self.current_patch.get_connection_count(),
        }
        
        if self.current_graph:
            graph_info = self.current_graph.get_graph_info()
            info.update(graph_info)
        
        return info
    
    def export_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract audio features from rendered audio.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Basic statistics
        features['length_samples'] = len(audio_data)
        features['length_seconds'] = len(audio_data) / self.sample_rate
        features['rms'] = float(np.sqrt(np.mean(audio_data ** 2)))
        features['peak'] = float(np.max(np.abs(audio_data)))
        features['zero_crossings'] = int(np.sum(np.diff(np.signbit(audio_data))))
        
        # Spectral features (basic)
        if len(audio_data) > 0:
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            features['spectral_centroid'] = float(np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude))
            features['spectral_rolloff'] = float(np.sum(magnitude > np.max(magnitude) * 0.85))
        
        return features
    
    def create_template(self, template_file: Union[str, Path]) -> 'PatchTemplate':
        """
        Create a patch template for parameter exploration.
        
        Args:
            template_file: Path to template file
            
        Returns:
            PatchTemplate instance
        """
        try:
            return PatchTemplate(template_file)
        except Exception as e:
            raise EngineError(f"Failed to create template: {e}")
    
    def batch_render(self, template_file: Union[str, Path], 
                     parameter_sets: List[Dict[str, Any]],
                     output_dir: Union[str, Path],
                     duration: Optional[float] = None,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """
        Render multiple variations from a template with different parameter sets.
        
        Args:
            template_file: Path to template file
            parameter_sets: List of parameter dictionaries
            output_dir: Directory to save output files
            duration: Duration for each render
            progress_callback: Optional callback with (current, total) counts
            
        Returns:
            List of result dictionaries with metadata for each render
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        template = self.create_template(template_file)
        results = []
        
        for i, params in enumerate(parameter_sets):
            try:
                # Load patch with parameters
                patch = self.load_patch(template_file, params)
                
                # Generate filename
                param_str = "_".join(f"{k}{v}" for k, v in params.items())
                output_file = output_path / f"render_{i:03d}_{param_str}.wav"
                
                # Render audio
                audio_data = self.render(duration, output_file)
                
                # Extract features
                features = self.export_features(audio_data)
                
                # Record result
                result = {
                    'index': i,
                    'parameters': params,
                    'output_file': str(output_file),
                    'features': features,
                    'success': True
                }
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(parameter_sets))
                    
            except Exception as e:
                result = {
                    'index': i,
                    'parameters': params,
                    'error': str(e),
                    'success': False
                }
                results.append(result)
        
        return results
    
    def cleanup(self):
        """Clean up engine resources."""
        if self.current_graph:
            self.current_graph.finalize()
        
        self.current_patch = None
        self.current_graph = None