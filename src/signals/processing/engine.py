"""
Main synthesizer engine providing high-level API for patch processing.

This module provides the SynthEngine class which serves as the main interface
for loading patches, processing audio, and managing the synthesis pipeline.
Designed for both interactive use and external program integration.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import time

from .patch import Patch, PatchTemplate, PatchError
from .graph import ModuleGraph, GraphError
from ..core.module import Signal, SignalType
from ..core.context import SynthContext, synthesis_context
from ..core.logging import get_logger, performance_logger, log_module_state


class EngineError(Exception):
    """Base exception for engine-related errors."""
    pass


class SynthEngine:
    """
    High-level synthesizer engine for processing patches and generating audio.
    
    The SynthEngine provides a simple interface for loading patch files,
    setting parameters dynamically, and rendering audio output. It handles
    all the complexity of module graph management and signal routing.
    
    The engine also serves as a synthesis context, allowing modules to be
    created without explicit sample rate specification.
    
    Example:
        >>> engine = SynthEngine(sample_rate=48000)
        >>> patch = engine.load_patch("synth.yaml")
        >>> audio_data = engine.render(duration=2.0)
        >>> engine.save_audio("output.wav", audio_data)
        
        >>> # Or use as context manager
        >>> with SynthEngine(sample_rate=48000) as engine:
        ...     # Modules created here automatically use engine's sample rate
        ...     osc = Oscillator()
        ...     env = EnvelopeADSR()
    """
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.current_patch: Optional[Patch] = None
        self.current_graph: Optional[ModuleGraph] = None
        self._processing_callbacks: List[Callable] = []
        self._context: Optional[SynthContext] = None
        self.logger = get_logger('processing.engine')
        
        self.logger.info(f"SynthEngine initialized: sample_rate={sample_rate}Hz, buffer_size={buffer_size}")
    
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
                self.logger.info(f"Loading patch template: {patch_path} with variables: {variables}")
                template = PatchTemplate(patch_path)
                patch = template.instantiate(variables)
            else:
                # Load as regular patch
                self.logger.info(f"Loading patch: {patch_path}")
                patch = Patch.from_file(patch_path)
            
            # Override sample rate from engine
            patch.sample_rate = self.sample_rate
            
            # Build module graph within context
            with self.context():
                self.current_patch = patch
                self.current_graph = ModuleGraph(patch)
            
            # Configure envelopes with auto duration detection
            self._configure_envelope_durations()
            
            self.logger.info(f"Patch loaded successfully: {patch.name} ({patch.get_module_count()} modules, {patch.get_connection_count()} connections)")
            
            return patch
            
        except (PatchError, GraphError) as e:
            self.logger.error(f"Failed to load patch: {e}")
            raise EngineError(f"Failed to load patch: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading patch: {e}")
            raise EngineError(f"Unexpected error loading patch: {e}")
    
    def __enter__(self):
        """Enter the engine as a context manager."""
        self._context = SynthContext(self.sample_rate)
        self._context.__enter__()
        self.logger.debug(f"Entered SynthEngine context: sample_rate={self.sample_rate}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the engine context manager."""
        if self._context:
            self._context.__exit__(exc_type, exc_val, exc_tb)
            self._context = None
        self.logger.debug("Exited SynthEngine context")
    
    @contextmanager
    def context(self):
        """
        Create a synthesis context with this engine's sample rate.
        
        This context manager allows modules to be created without explicit
        sample rate specification, using the engine's sample rate instead.
        
        Yields:
            SynthContext: Context with engine's sample rate
            
        Example:
            >>> engine = SynthEngine(sample_rate=48000)
            >>> with engine.context():
            ...     osc = Oscillator()  # Uses 48000 Hz
            ...     env = EnvelopeADSR()  # Uses 48000 Hz
        """
        with synthesis_context(self.sample_rate) as ctx:
            self.logger.debug(f"Created synthesis context: sample_rate={self.sample_rate}")
            yield ctx
    
    def load_patch_from_dict(self, patch_data: Dict[str, Any]) -> Patch:
        """
        Load patch from dictionary data.
        
        Args:
            patch_data: Dictionary containing patch configuration
            
        Returns:
            Loaded and validated Patch instance
        """
        try:
            self.logger.info("Loading patch from dictionary data")
            patch = Patch.from_dict(patch_data)
            patch.sample_rate = self.sample_rate
            
            self.current_patch = patch
            self.current_graph = ModuleGraph(patch)
            
            # Configure envelopes with auto duration detection
            self._configure_envelope_durations()
            
            self.logger.info(f"Patch loaded from dict: {patch.name} ({patch.get_module_count()} modules)")
            
            return patch
            
        except (PatchError, GraphError) as e:
            self.logger.error(f"Failed to load patch from dict: {e}")
            raise EngineError(f"Failed to load patch from dict: {e}")
    
    @performance_logger
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
            self.logger.error("No patch loaded. Call load_patch() first.")
            raise EngineError("No patch loaded. Call load_patch() first.")
        
        try:
            # Determine duration
            if duration is None:
                sequence_duration = self.current_patch.get_duration()
                if sequence_duration == 0.0:
                    sequence_duration = 2.0  # Default duration
            
                # Calculate additional time needed for envelope release phases
                max_release_time = 0.0
                for module_id, module_data in self.current_patch.modules.items():
                    if module_data['type'] == 'envelope_adsr':
                        envelope_module = self.current_graph.get_module(module_id)
                        if envelope_module and hasattr(envelope_module, 'release_time'):
                            max_release_time = max(max_release_time, envelope_module.release_time)
            
                # Total duration includes sequence time plus envelope release completion
                duration = sequence_duration + max_release_time
                self.logger.info(f"Auto-calculated duration: {duration:.3f}s (sequence: {sequence_duration:.3f}s + release: {max_release_time:.3f}s)")
            else:
                self.logger.info(f"Using specified duration: {duration:.3f}s")
            
            # Process the graph
            self.logger.debug(f"Starting audio rendering for {duration:.3f}s ({int(duration * self.sample_rate)} samples)")
            all_outputs = self.current_graph.process_duration(duration, progress_callback)
            
            # Extract audio from output modules
            audio_data = self._extract_audio_output(all_outputs, duration)
            
            # Save to file if requested
            if output_file:
                self.logger.info(f"Saving audio to: {output_file}")
                self.save_audio(output_file, audio_data)
            
            # Finalize modules
            self.current_graph.finalize()
            
            self.logger.info(f"Rendering completed: {len(audio_data)} samples generated")
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Rendering failed: {e}")
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
        from ..core import write_wav
        
        try:
            self.logger.debug(f"Saving {len(audio_data)} samples to {filename} ({bits_per_sample}-bit)")
            write_wav(str(filename), audio_data, self.sample_rate, bits_per_sample)
            self.logger.info(f"Audio saved successfully: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
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
            self.logger.error("No patch loaded. Call load_patch() first.")
            raise EngineError("No patch loaded. Call load_patch() first.")
        
        try:
            self.logger.debug(f"Setting parameter: {module_id}.{param_name} = {value}")
            self.current_graph.set_module_parameter(module_id, param_name, value)
        except Exception as e:
            self.logger.error(f"Failed to set parameter {module_id}.{param_name}: {e}")
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
    
    def _configure_envelope_durations(self):
        """Configure envelope modules with auto duration detection."""
        if not self.current_patch or not self.current_graph:
            return
        
        # Detect total duration from patch sequence
        total_duration = self.current_patch.get_duration()
        if total_duration == 0.0:
            total_duration = 2.0  # Default duration
        
        # Find the maximum release time needed and extend total duration accordingly
        max_release_extension = 0.0
        envelope_count = 0
        
        # First pass: calculate envelope durations to determine maximum release extension
        for module_id, module_data in self.current_patch.modules.items():
            if module_data['type'] == 'envelope_adsr':
                envelope_count += 1
                # Calculate release time based on parameters
                params = module_data['parameters']
                release_param = params.get('release', 0.2)
                
                if release_param == "auto":
                    # For auto release, estimate based on reasonable default
                    estimated_release = total_duration * 0.3  # 30% of total
                elif isinstance(release_param, str) and release_param.endswith('%'):
                    percentage = float(release_param[:-1])
                    estimated_release = total_duration * (percentage / 100.0)
                else:
                    estimated_release = float(release_param)
                
                max_release_extension = max(max_release_extension, estimated_release)
        
        # Extend total duration to include release completion
        extended_duration = total_duration + max_release_extension
        
        if envelope_count > 0:
            self.logger.info(f"Configuring {envelope_count} envelope(s): base_duration={total_duration:.3f}s, "
                           f"max_release={max_release_extension:.3f}s, extended_duration={extended_duration:.3f}s")
        
        # Configure all envelope modules with extended duration
        for module_id, module_data in self.current_patch.modules.items():
            if module_data['type'] == 'envelope_adsr':
                envelope_module = self.current_graph.get_module(module_id)
                if envelope_module and hasattr(envelope_module, 'set_total_duration'):
                    envelope_module.set_total_duration(extended_duration)
                    self.logger.debug(f"Configured envelope {module_id} with duration {extended_duration:.3f}s")
    
    def cleanup(self):
        """Clean up engine resources."""
        if self.current_graph:
            self.current_graph.finalize()
        
        self.current_patch = None
        self.current_graph = None