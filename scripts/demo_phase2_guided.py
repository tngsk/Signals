#!/usr/bin/env python3
"""
Guided Phase 2 Demo for Signals

This script provides a step-by-step demonstration of Phase 2 features
without requiring interactive input. Perfect for understanding the
implementation through automated examples.

Run with: uv run python demo_phase2_guided.py
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from signals import SynthEngine, Patch, PatchTemplate, ModuleGraph
    from signals.processing.patch import PatchError, Connection, SequenceEvent
    from signals.processing.engine import EngineError
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the Signals directory and run with: uv run python scripts/demo_phase2_guided.py")
    sys.exit(1)


class GuidedDemo:
    """Guided demonstration of Phase 2 features."""
    
    def __init__(self):
        self.step = 0
        self.engine = None
        self.current_patch = None
        self.created_files = []
    
    def print_header(self, title: str):
        """Print a formatted section header."""
        print("\n" + "=" * 60)
        print(f"  📚 {title}")
        print("=" * 60)
    
    def print_step(self, title: str):
        """Print a step header."""
        self.step += 1
        print(f"\n🔹 Step {self.step}: {title}")
        print("-" * 40)
    
    def pause(self, seconds: float = 1.0):
        """Add a short pause for readability."""
        time.sleep(seconds)
    
    def show_code(self, code: str, description: str = "Code example:"):
        """Display code with syntax highlighting simulation."""
        print(f"\n📝 {description}")
        print("```python")
        for line in code.strip().split('\n'):
            print(f"  {line}")
        print("```")
    
    def execute_and_show(self, code: str, description: str = "Executing:"):
        """Execute code and show the result."""
        print(f"\n⚡ {description}")
        print(f">>> {code}")
        try:
            result = eval(code)
            if result is not None:
                print(f"    {result}")
            return result
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def create_demo_file(self, filename: str, content: str, description: str):
        """Create a demo file and track it."""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        self.created_files.append(filepath)
        print(f"📄 Created {description}: {filename}")
    
    def cleanup(self):
        """Clean up created files."""
        for filepath in self.created_files:
            if filepath.exists():
                filepath.unlink()
                print(f"🗑️  Cleaned up: {filepath}")
        if self.engine:
            self.engine.cleanup()

    def demo_1_basic_concepts(self):
        """Demo 1: Basic Phase 2 concepts."""
        self.print_header("Demo 1: Understanding Phase 2 Concepts")
        
        print("""
🎯 Phase 2 introduces three major components:

1. 📋 Patch System - Load synthesizer configurations from YAML files
2. 🔗 Module Graph - Automatic signal routing and execution ordering  
3. 🎛️ SynthEngine - High-level API for synthesis control

Let's explore each component step by step!
        """)
        
        self.pause(2.0)
        
        # Show the architecture
        print("""
🏗️  Phase 2 Architecture:

    YAML Patch File
         ↓
    Patch Object (validated)
         ↓  
    Module Graph (connected modules)
         ↓
    SynthEngine (high-level control)
         ↓
    Audio Output
        """)
        
        self.pause(2.0)

    def demo_2_patch_basics(self):
        """Demo 2: Patch system basics."""
        self.print_header("Demo 2: Patch System Basics")
        
        self.print_step("Understanding Patch Structure")
        
        # Create a simple patch file
        simple_patch = """name: "My First Patch"
description: "A simple oscillator patch"
sample_rate: 48000

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: 440.0
      waveform: "sine"
      amplitude: 0.8

  env1:
    type: "envelope_adsr"
    parameters:
      attack: 0.05
      decay: 0.3
      sustain: 0.5
      release: 0.2

connections:
  - from: "osc1.0"
    to: "env1.0"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 1.5
    action: "release"
    target: "env1"
"""
        
        self.create_demo_file("demo_simple.yaml", simple_patch, "simple patch file")
        
        print("""
📋 A patch file contains:
- 📝 Metadata (name, description, sample_rate)
- 🔧 Modules (synthesizer components with parameters)
- 🔗 Connections (how modules connect to each other)
- ⏰ Sequence (timed events for automation)
        """)
        
        self.pause(1.5)
        
        self.print_step("Loading a Patch")
        
        self.show_code("""
# Create a Patch object from YAML file
from signals import Patch

patch = Patch.from_file("demo_simple.yaml")
print(f"Loaded: {patch.name}")
print(f"Modules: {list(patch.modules.keys())}")
print(f"Connections: {len(patch.connections)}")
        """)
        
        # Execute the code
        try:
            patch = Patch.from_file("demo_simple.yaml")
            print(f"\n✅ Loaded: {patch.name}")
            print(f"   Modules: {list(patch.modules.keys())}")
            print(f"   Connections: {len(patch.connections)}")
            print(f"   Sequence events: {len(patch.sequence)}")
            self.current_patch = patch
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(1.5)
        
        self.print_step("Examining Patch Contents")
        
        if self.current_patch:
            print("🔍 Let's examine the patch structure:")
            
            print(f"\n📦 Modules ({len(self.current_patch.modules)}):")
            for module_id, module_data in self.current_patch.modules.items():
                print(f"  - {module_id}: {module_data['type']}")
                for param, value in module_data['parameters'].items():
                    print(f"    {param}: {value}")
            
            print(f"\n🔗 Connections ({len(self.current_patch.connections)}):")
            for i, conn in enumerate(self.current_patch.connections):
                print(f"  {i+1}. {conn.source_module}.{conn.source_output} → {conn.dest_module}.{conn.dest_input}")
            
            print(f"\n⏰ Sequence ({len(self.current_patch.sequence)}):")
            for event in self.current_patch.sequence:
                print(f"  {event.time}s: {event.action} → {event.target}")
        
        self.pause(2.0)

    def demo_3_synthengine_basics(self):
        """Demo 3: SynthEngine basics."""
        self.print_header("Demo 3: SynthEngine - The Control Center")
        
        self.print_step("Creating a SynthEngine")
        
        self.show_code("""
# SynthEngine is the main interface for synthesis
from signals import SynthEngine

engine = SynthEngine(sample_rate=48000)
print(f"Engine created with sample rate: {engine.sample_rate}")
        """)
        
        self.engine = SynthEngine(sample_rate=48000)
        print(f"\n✅ Engine created with sample rate: {self.engine.sample_rate}")
        
        self.pause(1.0)
        
        self.print_step("Loading a Patch into the Engine")
        
        self.show_code("""
# Load our patch into the engine
patch = engine.load_patch("demo_simple.yaml")
print(f"Loaded patch: {patch.name}")

# Get patch information
info = engine.get_patch_info()
print(f"Modules: {info['modules']}")
print(f"Duration: {info['duration']}s")
        """)
        
        try:
            patch = self.engine.load_patch("demo_simple.yaml")
            print(f"\n✅ Loaded patch: {patch.name}")
            
            info = self.engine.get_patch_info()
            print(f"   Modules: {info['modules']}")
            print(f"   Duration: {info['duration']}s")
            print(f"   Sample rate: {info['sample_rate']} Hz")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(1.5)
        
        self.print_step("Rendering Audio")
        
        print("🎵 Now let's generate some audio!")
        
        self.show_code("""
# Render audio from the patch
audio_data = engine.render(duration=1.0)
print(f"Generated {len(audio_data)} samples")

# Extract audio features
features = engine.export_features(audio_data)
print(f"Peak level: {features['peak']:.3f}")
print(f"RMS level: {features['rms']:.3f}")
        """)
        
        try:
            print("\n🎵 Rendering audio...")
            audio_data = self.engine.render(duration=1.0)
            print(f"✅ Generated {len(audio_data)} samples")
            
            features = self.engine.export_features(audio_data)
            print(f"   Peak level: {features['peak']:.3f}")
            print(f"   RMS level: {features['rms']:.3f}")
            print(f"   Length: {features['length_seconds']:.2f} seconds")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(2.0)

    def demo_4_dynamic_control(self):
        """Demo 4: Dynamic parameter control."""
        self.print_header("Demo 4: Dynamic Parameter Control")
        
        self.print_step("Examining Current Parameters")
        
        if self.engine and self.engine.current_patch:
            print("🔍 Current module parameters:")
            
            try:
                for module_id in self.engine.current_patch.modules.keys():
                    params = self.engine.get_module_parameters(module_id)
                    print(f"\n📦 {module_id}:")
                    for param, value in params.items():
                        print(f"   {param}: {value}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        self.pause(1.5)
        
        self.print_step("Changing Parameters Dynamically")
        
        self.show_code("""
# Change oscillator frequency
engine.set_module_parameter("osc1", "frequency", 880.0)
print("Changed frequency to 880 Hz")

# Change envelope attack time
engine.set_module_parameter("env1", "attack", 0.1)
print("Changed attack time to 0.1 seconds")
        """)
        
        try:
            self.engine.set_module_parameter("osc1", "frequency", 880.0)
            print("✅ Changed frequency to 880 Hz")
            
            self.engine.set_module_parameter("env1", "attack", 0.1)
            print("✅ Changed attack time to 0.1 seconds")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(1.0)
        
        self.print_step("Testing the Changes")
        
        print("🎵 Rendering audio with the new parameters:")
        
        try:
            audio_data = self.engine.render(duration=0.5)
            features = self.engine.export_features(audio_data)
            print(f"✅ New audio: peak={features['peak']:.3f}, rms={features['rms']:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(1.5)

    def demo_5_templates(self):
        """Demo 5: Template system."""
        self.print_header("Demo 5: Template System for Parameter Exploration")
        
        self.print_step("Understanding Templates")
        
        print("""
🎯 Templates allow you to create parameterized patches where you can:
- Define variable placeholders in YAML files
- Generate multiple variations with different parameters
- Enable systematic parameter space exploration
        """)
        
        # Create a template file
        template_content = """name: "Template Demo"
description: "Parameterized patch for exploration"

variables:
  osc_freq: 440.0
  osc_wave: "sine"
  env_attack: 0.02

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: {{ osc_freq | default(440.0) }}
      waveform: "{{ osc_wave | default('sine') }}"
      amplitude: 0.8

  env1:
    type: "envelope_adsr"
    parameters:
      attack: {{ env_attack | default(0.02) }}
      decay: 0.3
      sustain: 0.5
      release: 0.2

connections:
  - from: "osc1.0"
    to: "env1.0"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 1.0
    action: "release"
    target: "env1"
"""
        
        self.create_demo_file("demo_template.yaml", template_content, "template file")
        
        self.pause(1.0)
        
        self.print_step("Creating a Template Object")
        
        self.show_code("""
# Create a template from the file
from signals import PatchTemplate

template = PatchTemplate("demo_template.yaml")
print(f"Template variables: {template.variables}")

# Get variable schema (default values)
schema = template.get_variable_schema()
print(f"Default values: {schema}")
        """)
        
        try:
            template = PatchTemplate("demo_template.yaml")
            print(f"\n✅ Template variables: {template.variables}")
            
            schema = template.get_variable_schema()
            print(f"   Default values: {schema}")
        except Exception as e:
            print(f"❌ Error: {e}")
            template = None
        
        self.pause(1.5)
        
        self.print_step("Generating Variations")
        
        if template:
            print("🎵 Creating different variations:")
            
            variations = [
                {"osc_freq": 220.0, "osc_wave": "sine", "env_attack": 0.01},
                {"osc_freq": 440.0, "osc_wave": "square", "env_attack": 0.05},
                {"osc_freq": 880.0, "osc_wave": "triangle", "env_attack": 0.1}
            ]
            
            for i, params in enumerate(variations, 1):
                print(f"\n🔸 Variation {i}: {params}")
                
                try:
                    # Load patch with parameters
                    patch = self.engine.load_patch("demo_template.yaml", params)
                    print(f"   ✅ Loaded: {patch.name}")
                    
                    # Quick render
                    audio = self.engine.render(duration=0.3)
                    features = self.engine.export_features(audio)
                    print(f"   📊 Peak: {features['peak']:.3f}, RMS: {features['rms']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                
                self.pause(0.5)
        
        self.pause(1.5)

    def demo_6_module_graph(self):
        """Demo 6: Module graph system."""
        self.print_header("Demo 6: Module Graph - The Signal Routing Engine")
        
        self.print_step("Understanding the Module Graph")
        
        print("""
🔗 The Module Graph automatically:
- Creates instances of all modules from the patch
- Connects modules according to the connections list
- Determines the correct execution order (topological sorting)
- Handles signal routing between modules
        """)
        
        # Create a more complex patch
        complex_patch = """name: "Complex Graph Demo"
description: "Multi-module patch to demonstrate graph features"

modules:
  osc1:
    type: "oscillator"
    parameters:
      frequency: 440.0
      waveform: "sine"

  osc2:
    type: "oscillator"
    parameters:
      frequency: 660.0
      waveform: "square"

  mixer:
    type: "mixer"
    parameters:
      num_inputs: 2
      gain1: 0.7
      gain2: 0.3

  env1:
    type: "envelope_adsr"
    parameters:
      attack: 0.1
      decay: 0.5
      sustain: 0.3
      release: 0.3

connections:
  - from: "osc1.0"
    to: "mixer.0"
  - from: "osc2.0"
    to: "mixer.1"
  - from: "mixer.0"
    to: "env1.0"

sequence:
  - time: 0.0
    action: "trigger"
    target: "env1"
  - time: 2.0
    action: "release"
    target: "env1"
"""
        
        self.create_demo_file("demo_complex.yaml", complex_patch, "complex patch file")
        
        self.pause(1.0)
        
        self.print_step("Loading and Examining the Graph")
        
        try:
            patch = self.engine.load_patch("demo_complex.yaml")
            print(f"✅ Loaded: {patch.name}")
            
            info = self.engine.get_patch_info()
            print(f"   Modules: {info['modules']}")
            print(f"   Connections: {info['connection_count']}")
            
            if 'execution_order' in info:
                print(f"   Execution order: {info['execution_order']}")
                
                print("\n🔍 Execution order explanation:")
                for i, module_id in enumerate(info['execution_order']):
                    module_type = patch.modules[module_id]['type']
                    print(f"   {i+1}. {module_id} ({module_type})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(1.5)
        
        self.print_step("Understanding Signal Flow")
        
        print("""
🌊 Signal flow in this patch:

   osc1 (440Hz sine) ────┐
                         ├─→ mixer ─→ env1 ─→ output
   osc2 (660Hz square) ──┘

1. Both oscillators generate audio simultaneously
2. Mixer combines them with different gain levels (70% + 30%)
3. Envelope shapes the mixed signal
4. Final audio is output
        """)
        
        self.pause(2.0)
        
        self.print_step("Rendering Complex Audio")
        
        try:
            print("🎵 Rendering the complex patch...")
            audio = self.engine.render(duration=1.0)
            features = self.engine.export_features(audio)
            
            print(f"✅ Complex audio generated:")
            print(f"   Length: {features['length_seconds']:.2f}s")
            print(f"   Peak: {features['peak']:.3f}")
            print(f"   RMS: {features['rms']:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.pause(1.5)

    def demo_7_advanced_features(self):
        """Demo 7: Advanced features."""
        self.print_header("Demo 7: Advanced Features for External Control")
        
        self.print_step("Batch Processing Concept")
        
        print("""
🔄 Phase 2 enables batch processing for parameter exploration:
- Load templates with different parameter sets
- Generate multiple audio files automatically
- Extract features for analysis
        """)
        
        self.show_code("""
# Example batch processing code structure
parameter_sets = [
    {"osc_freq": 220.0, "env_attack": 0.01},
    {"osc_freq": 440.0, "env_attack": 0.05},
    {"osc_freq": 880.0, "env_attack": 0.1}
]

# Loop through and generate variations
# for params in parameter_sets:
#     patch = engine.load_patch(template_file, params)
#     engine.render(output_file=...)
        """)
        
        self.pause(1.5)
        
        self.print_step("Programmatic Control")
        
        print("""
🎛️ Perfect for external programs:

1. Python API - Direct control from scripts
2. Template system - Systematic parameter exploration  
3. Feature extraction - Automatic analysis
4. Batch processing - High-throughput generation
        """)
        
        self.show_code("""
# Example external control workflow:

# 1. Load template
engine = SynthEngine()
template = engine.create_template("synth_template.yaml")

# 2. Generate variations
for params in parameter_space:
    patch = engine.load_patch(template_file, params)
    audio = engine.render(duration=2.0)
    features = engine.export_features(audio)
    
    # 3. Save results for analysis
    save_results(params, audio, features)
        """)
        
        self.pause(1.5)
        
        self.print_step("Integration Points")
        
        print("""
🔗 Phase 2 is designed for integration with:

✅ Machine Learning pipelines
✅ Parameter optimization algorithms  
✅ Audio analysis frameworks
✅ Procedural music generation
✅ Research experiments
✅ Automated composition systems

The modular design makes it easy to:
- Swap different modules
- Add new synthesis methods
- Customize parameter ranges
- Extract custom features
        """)
        
        self.pause(2.0)

    def demo_8_practical_example(self):
        """Demo 8: Practical example."""
        self.print_header("Demo 8: Practical Example - Parameter Exploration")
        
        self.print_step("Simulating External Control")
        
        print("🔬 Let's simulate how an external program would use Signals:")
        
        # Simulate a parameter exploration
        print("\n🎯 Exploring frequency and envelope parameters:")
        
        frequencies = [220.0, 440.0, 880.0]
        attack_times = [0.01, 0.05, 0.1]
        
        results = []
        
        for freq in frequencies:
            for attack in attack_times:
                params = {
                    "osc_freq": freq,
                    "osc_wave": "sine",
                    "env_attack": attack
                }
                
                try:
                    print(f"\n📊 Testing: {freq}Hz, attack={attack}s")
                    
                    # Load patch with parameters
                    patch = self.engine.load_patch("demo_template.yaml", params)
                    
                    # Quick render
                    audio = self.engine.render(duration=0.5)
                    features = self.engine.export_features(audio)
                    
                    result = {
                        'frequency': freq,
                        'attack': attack,
                        'peak': features['peak'],
                        'rms': features['rms']
                    }
                    results.append(result)
                    
                    print(f"   ✅ Peak: {features['peak']:.3f}, RMS: {features['rms']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                
                self.pause(0.3)
        
        self.pause(1.0)
        
        self.print_step("Analysis Results")
        
        print("\n📈 Parameter exploration results:")
        print(f"{'Freq (Hz)':>8} {'Attack (s)':>10} {'Peak':>8} {'RMS':>8}")
        print("-" * 36)
        
        for result in results:
            print(f"{result['frequency']:>8.0f} {result['attack']:>10.3f} "
                  f"{result['peak']:>8.3f} {result['rms']:>8.3f}")
        
        self.pause(1.5)

    def demo_9_summary(self):
        """Demo 9: Summary and next steps."""
        self.print_header("Demo 9: Summary and Next Steps")
        
        print("""
🎉 You've seen all major Phase 2 features in action:

✅ Patch System
  - YAML-based configuration files
  - Automatic validation and error checking
  - Support for modules, connections, and sequences

✅ Template System  
  - Jinja2-powered parameter substitution
  - Systematic variation generation
  - Default value handling

✅ SynthEngine API
  - High-level synthesis control
  - Dynamic parameter modification
  - Audio feature extraction

✅ Module Graph
  - Automatic signal routing
  - Execution order optimization
  - Complex module networks

✅ External Integration
  - Programmatic Python API
  - Batch processing capabilities
  - Ready for parameter exploration
        """)
        
        self.pause(1.5)
        
        print("""
🚀 What you can do now:

1. Create your own patch files using the examples as templates
2. Build parameterized templates for systematic exploration
3. Use the SynthEngine in your own Python projects
4. Integrate with external optimization/ML programs
5. Explore complex parameter spaces efficiently

📁 Files created during this demo:
        """)
        
        for filepath in self.created_files:
            print(f"   - {filepath}")
        
        print("""
💡 Try these next steps:

1. Edit the generated patch files and experiment
2. Create templates with different parameter ranges
3. Add new module types to your patches
4. Build more complex signal routing graphs
5. Integrate with your own analysis pipelines

🔧 Key classes to remember:
- SynthEngine: Main control interface
- Patch: Configuration container
- PatchTemplate: Parameterized patches
- ModuleGraph: Signal routing engine

Happy synthesizing! 🎵✨
        """)

    def run_demo(self):
        """Run the complete guided demonstration."""
        print("""
🎵 Welcome to the Signals Phase 2 Guided Demo!

This demo will show you all the new features through working examples.
No interaction required - just sit back and watch the code in action!
        """)
        
        self.pause(2.0)
        
        try:
            self.demo_1_basic_concepts()
            self.demo_2_patch_basics()
            self.demo_3_synthengine_basics()
            self.demo_4_dynamic_control()
            self.demo_5_templates()
            self.demo_6_module_graph()
            self.demo_7_advanced_features()
            self.demo_8_practical_example()
            self.demo_9_summary()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up...")
            self.cleanup()
            print("\n🎉 Thanks for watching the Phase 2 demo! 👋")


def main():
    """Main entry point."""
    demo = GuidedDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()