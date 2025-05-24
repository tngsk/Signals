"""
Phase 2 Demo: Patch System and SynthEngine

This script demonstrates the new Phase 2 functionality including:
- Loading patches from YAML files
- Using parameterized templates 
- Dynamic parameter control
- Batch processing with variations
- Audio feature extraction
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from signals import SynthEngine, PatchTemplate
from signals.patch import Patch


def demo_basic_patch():
    """Demonstrate basic patch loading and rendering."""
    print("=== Demo 1: Basic Patch Loading ===")
    
    engine = SynthEngine(sample_rate=48000)
    
    # Create a simple patch programmatically
    patch_data = {
        'name': 'Demo Basic Synth',
        'description': 'Simple oscillator with envelope',
        'modules': {
            'osc1': {
                'type': 'oscillator',
                'parameters': {
                    'frequency': 440.0,
                    'waveform': 'sine',
                    'amplitude': 0.7
                }
            },
            'env1': {
                'type': 'envelope_adsr',
                'parameters': {
                    'attack': 0.05,
                    'decay': 0.3,
                    'sustain': 0.4,
                    'release': 0.2
                }
            },
            'output': {
                'type': 'output_wav',
                'parameters': {
                    'filename': 'demo_basic.wav'
                }
            }
        },
        'connections': [
            {'from': 'osc1.0', 'to': 'env1.0'},
            {'from': 'env1.0', 'to': 'output.0'}
        ],
        'sequence': [
            {'time': 0.0, 'action': 'trigger', 'target': 'env1'},
            {'time': 1.5, 'action': 'release', 'target': 'env1'}
        ]
    }
    
    # Load and render
    patch = engine.load_patch_from_dict(patch_data)
    print(f"Loaded patch: {patch.name}")
    print(f"Modules: {list(patch.modules.keys())}")
    print(f"Connections: {len(patch.connections)}")
    
    # Render audio
    print("Rendering audio...")
    audio_data = engine.render(duration=2.0)
    
    # Extract features
    features = engine.export_features(audio_data)
    print(f"Generated {features['length_seconds']:.2f}s of audio")
    print(f"Peak level: {features['peak']:.3f}")
    print(f"RMS level: {features['rms']:.3f}")
    
    engine.cleanup()
    print("✓ Basic patch demo complete\n")


def demo_patch_file():
    """Demonstrate loading from YAML patch file."""
    print("=== Demo 2: Loading from YAML File ===")
    
    engine = SynthEngine(sample_rate=48000)
    patch_file = Path("examples/patches/basic_synth.yaml")
    
    if patch_file.exists():
        try:
            patch = engine.load_patch(patch_file)
            print(f"Loaded patch: {patch.name}")
            print(f"Description: {patch.description}")
            
            # Get patch info
            info = engine.get_patch_info()
            print(f"Sample rate: {info['sample_rate']} Hz")
            print(f"Modules: {info['modules']}")
            print(f"Duration: {info['duration']}s")
            
            # Render
            audio_data = engine.render()
            print(f"Rendered {len(audio_data)} samples")
            
        except Exception as e:
            print(f"Error loading patch file: {e}")
    else:
        print(f"Patch file not found: {patch_file}")
    
    engine.cleanup()
    print("✓ Patch file demo complete\n")


def demo_template_system():
    """Demonstrate template system with parameters."""
    print("=== Demo 3: Template System ===")
    
    engine = SynthEngine(sample_rate=48000)
    template_file = Path("examples/templates/parametric_synth.yaml")
    
    if template_file.exists():
        try:
            # Load template
            template = engine.create_template(template_file)
            print(f"Template variables: {template.variables}")
            
            # Generate variations
            variations = [
                {
                    'osc_freq': 220.0,
                    'osc_waveform': 'sine',
                    'env_attack': 0.01,
                    'output_filename': 'variation_220hz.wav'
                },
                {
                    'osc_freq': 440.0,
                    'osc_waveform': 'square', 
                    'env_attack': 0.05,
                    'output_filename': 'variation_440hz.wav'
                },
                {
                    'osc_freq': 880.0,
                    'osc_waveform': 'triangle',
                    'env_attack': 0.1,
                    'output_filename': 'variation_880hz.wav'
                }
            ]
            
            results = []
            for i, params in enumerate(variations):
                print(f"Generating variation {i+1}: {params['osc_freq']}Hz {params['osc_waveform']}")
                
                # Load patch with parameters
                patch = engine.load_patch(template_file, params)
                
                # Render
                audio_data = engine.render(duration=1.0)
                features = engine.export_features(audio_data)
                
                result = {
                    'frequency': params['osc_freq'],
                    'waveform': params['osc_waveform'], 
                    'peak': features['peak'],
                    'rms': features['rms']
                }
                results.append(result)
                
            # Show results
            print("\nVariation Results:")
            for result in results:
                print(f"  {result['frequency']:>6.0f}Hz {result['waveform']:>8s}: "
                      f"peak={result['peak']:.3f}, rms={result['rms']:.3f}")
                
        except Exception as e:
            print(f"Error with template: {e}")
    else:
        print(f"Template file not found: {template_file}")
    
    engine.cleanup()
    print("✓ Template system demo complete\n")


def demo_dynamic_control():
    """Demonstrate dynamic parameter control during rendering."""
    print("=== Demo 4: Dynamic Parameter Control ===")
    
    engine = SynthEngine(sample_rate=48000)
    
    # Simple oscillator patch
    patch_data = {
        'name': 'Dynamic Control Demo',
        'modules': {
            'osc1': {
                'type': 'oscillator',
                'parameters': {'frequency': 440.0, 'waveform': 'sine'}
            }
        }
    }
    
    patch = engine.load_patch_from_dict(patch_data)
    
    # Show original parameters
    original_params = engine.get_module_parameters('osc1')
    print(f"Original frequency: {original_params['frequency']} Hz")
    
    # Change frequency dynamically
    new_frequency = 880.0
    engine.set_module_parameter('osc1', 'frequency', new_frequency)
    print(f"Changed frequency to: {new_frequency} Hz")
    
    # Quick render test
    audio_data = engine.render(duration=0.5)
    features = engine.export_features(audio_data)
    print(f"Rendered audio with peak: {features['peak']:.3f}")
    
    engine.cleanup()
    print("✓ Dynamic control demo complete\n")


def demo_multi_module_patch():
    """Demonstrate complex patch with multiple modules."""
    print("=== Demo 5: Multi-Module Patch ===")
    
    engine = SynthEngine(sample_rate=48000)
    
    # More complex patch with mixer
    patch_data = {
        'name': 'Multi-Module Demo',
        'modules': {
            'osc1': {
                'type': 'oscillator',
                'parameters': {'frequency': 440.0, 'waveform': 'sine'}
            },
            'osc2': {
                'type': 'oscillator', 
                'parameters': {'frequency': 660.0, 'waveform': 'square'}
            },
            'mixer': {
                'type': 'mixer',
                'parameters': {'gain1': 0.7, 'gain2': 0.3}
            },
            'env1': {
                'type': 'envelope_adsr',
                'parameters': {'attack': 0.1, 'decay': 0.5, 'sustain': 0.3, 'release': 0.3}
            }
        },
        'connections': [
            {'from': 'osc1.0', 'to': 'mixer.0'},
            {'from': 'osc2.0', 'to': 'mixer.1'},
            {'from': 'mixer.0', 'to': 'env1.0'}
        ],
        'sequence': [
            {'time': 0.0, 'action': 'trigger', 'target': 'env1'},
            {'time': 2.0, 'action': 'release', 'target': 'env1'}
        ]
    }
    
    patch = engine.load_patch_from_dict(patch_data)
    info = engine.get_patch_info()
    
    print(f"Loaded complex patch: {patch.name}")
    print(f"Module count: {info['module_count']}")
    print(f"Connection count: {info['connection_count']}")
    print(f"Execution order: {info.get('execution_order', 'N/A')}")
    
    # Render
    audio_data = engine.render(duration=3.0)
    features = engine.export_features(audio_data)
    
    print(f"Rendered {features['length_seconds']:.1f}s of audio")
    print(f"Peak level: {features['peak']:.3f}")
    print(f"Spectral centroid: {features.get('spectral_centroid', 'N/A')}")
    
    engine.cleanup()
    print("✓ Multi-module demo complete\n")


def main():
    """Run all Phase 2 demos."""
    print("🎵 Signals Phase 2 Demo")
    print("=" * 50)
    
    try:
        demo_basic_patch()
        demo_patch_file()
        demo_template_system()
        demo_dynamic_control()
        demo_multi_module_patch()
        
        print("🎉 All Phase 2 demos completed successfully!")
        print("\nPhase 2 Features Demonstrated:")
        print("✓ Patch loading from dictionaries and YAML files")
        print("✓ Template system with parameter substitution")
        print("✓ Dynamic parameter control")
        print("✓ Multi-module graphs with connections")
        print("✓ Audio feature extraction")
        print("✓ Sequence-based automation")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()