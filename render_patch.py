#!/usr/bin/env python3
"""
Render audio from patch files.

This script loads a patch file and generates audio output using the Signals engine.
Supports both basic patches and parameterized templates with variable substitution.

Usage:
    python render_patch.py examples/patches/basic_synth.yaml
    python render_patch.py examples/templates/parametric_synth.yaml --vars "osc_freq=880,env_attack=0.05"
    python render_patch.py examples/patches/basic_synth.yaml --duration 5.0 --output custom_output.wav
"""

import sys
import argparse
from pathlib import Path
import json

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from signals import SynthEngine, PatchTemplate
from signals.processing.patch import PatchError
from signals.processing.engine import EngineError


def parse_variables(vars_string):
    """Parse variable string in format 'key1=value1,key2=value2'."""
    if not vars_string:
        return {}
    
    variables = {}
    for pair in vars_string.split(','):
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert to appropriate type
        if value.lower() == 'true':
            variables[key] = True
        elif value.lower() == 'false':
            variables[key] = False
        elif value.replace('.', '').replace('-', '').isdigit():
            variables[key] = float(value) if '.' in value else int(value)
        else:
            variables[key] = value
    
    return variables


def main():
    parser = argparse.ArgumentParser(
        description='Render audio from Signals patch files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s examples/patches/basic_synth.yaml
  %(prog)s examples/templates/parametric_synth.yaml --vars "osc_freq=880,env_attack=0.05"
  %(prog)s examples/patches/basic_synth.yaml --duration 5.0 --output custom.wav
  %(prog)s examples/templates/parametric_synth.yaml --info --vars "osc_freq=440"
        """
    )
    
    parser.add_argument(
        'patch_file',
        help='Path to patch file (.yaml)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        help='Audio duration in seconds (default: use patch sequence duration or 2.0s)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output audio file path (default: use patch filename parameter or auto-generate)'
    )
    
    parser.add_argument(
        '--vars', '-v',
        help='Template variables in format "key1=value1,key2=value2"'
    )
    
    parser.add_argument(
        '--sample-rate', '-r',
        type=int,
        default=48000,
        help='Sample rate in Hz (default: 48000)'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show patch information without rendering'
    )
    
    parser.add_argument(
        '--features', '-f',
        action='store_true',
        help='Extract and display audio features after rendering'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output messages'
    )
    
    args = parser.parse_args()
    
    # Validate patch file
    patch_path = Path(args.patch_file)
    if not patch_path.exists():
        print(f"Error: Patch file '{patch_path}' not found", file=sys.stderr)
        return 1
    
    if not patch_path.suffix.lower() in ['.yaml', '.yml']:
        print(f"Error: Patch file must be a YAML file (.yaml or .yml)", file=sys.stderr)
        return 1
    
    try:
        # Create engine
        engine = SynthEngine(sample_rate=args.sample_rate)
        
        # Parse variables if provided
        variables = parse_variables(args.vars) if args.vars else None
        
        # Check if this is a template by looking for variables
        is_template = False
        if variables is not None or args.info:
            try:
                template = PatchTemplate(patch_path)
                is_template = True
                
                if args.info:
                    if not args.quiet:
                        print(f"Template: {patch_path}")
                        print(f"Available variables: {template.variables}")
                        schema = template.get_variable_schema()
                        if schema:
                            print("Default values:")
                            for var, default in schema.items():
                                print(f"  {var}: {default}")
                        else:
                            print("No default values defined")
                    
                    if not variables:
                        return 0
                        
            except Exception:
                # Not a template, continue as regular patch
                pass
        
        # Load patch
        if not args.quiet:
            action = "Loading template" if (is_template and variables) else "Loading patch"
            print(f"{action}: {patch_path}")
            if variables:
                print(f"Variables: {variables}")
        
        if variables:
            patch = engine.load_patch(patch_path, variables=variables)
        else:
            patch = engine.load_patch(patch_path)
        
        # Show patch info
        if args.info:
            info = engine.get_patch_info()
            if not args.quiet:
                print(f"\nPatch Information:")
                print(f"  Name: {info['name']}")
                print(f"  Description: {info.get('description', 'N/A')}")
                print(f"  Sample rate: {info['sample_rate']} Hz")
                print(f"  Modules: {info['modules']}")
                print(f"  Module count: {info['module_count']}")
                print(f"  Connection count: {info['connection_count']}")
                print(f"  Duration: {info['duration']}s")
                if 'execution_order' in info:
                    print(f"  Execution order: {info['execution_order']}")
            
            if args.duration is None:
                return 0
        
        # Determine duration
        duration = args.duration
        # If no duration specified, let engine calculate automatically (including envelope release)
        
        # Render audio
        if duration is not None:
            if not args.quiet:
                print(f"Rendering {duration}s of audio...")
        else:
            if not args.quiet:
                print(f"Rendering audio with automatic duration calculation...")
        
        output_file = args.output
        audio_data = engine.render(duration=duration, output_file=output_file)
        
        # Extract features if requested
        if args.features:
            features = engine.export_features(audio_data)
            if not args.quiet:
                print(f"\nAudio Features:")
                print(f"  Length: {features['length_seconds']:.2f}s")
                print(f"  Samples: {features['length_samples']}")
                print(f"  Peak level: {features['peak']:.3f}")
                print(f"  RMS level: {features['rms']:.3f}")
                print(f"  Zero crossings: {features['zero_crossings']}")
                if 'spectral_centroid' in features:
                    print(f"  Spectral centroid: {features['spectral_centroid']:.1f}")
                if 'spectral_rolloff' in features:
                    print(f"  Spectral rolloff: {features['spectral_rolloff']:.1f}")
        
        # Determine output filename for display
        if not output_file:
            # Look for output module in patch
            info = engine.get_patch_info()
            output_file = "Generated by patch output module"
        
        if not args.quiet:
            actual_duration = len(audio_data) / args.sample_rate
            print(f"✅ Audio rendered successfully")
            if output_file != "Generated by patch output module":
                print(f"   Output: {output_file}")
            print(f"   Duration: {actual_duration:.1f}s")
            print(f"   Samples: {len(audio_data)}")
            print(f"   Sample rate: {args.sample_rate} Hz")
        
        return 0
        
    except PatchError as e:
        print(f"Patch error: {e}", file=sys.stderr)
        return 1
    except EngineError as e:
        print(f"Engine error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"File error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if 'engine' in locals():
            engine.cleanup()


if __name__ == "__main__":
    sys.exit(main())