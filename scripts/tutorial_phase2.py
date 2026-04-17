#!/usr/bin/env python3
"""
Interactive Phase 2 Tutorial for Signals

This script provides a step-by-step, interactive tutorial to explore
all the new Phase 2 features including patch loading, templates,
module graphs, and the SynthEngine API.

Run with: uv run python tutorial_phase2.py
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from signals import SynthEngine, Patch, PatchTemplate, ModuleGraph
    from signals.patch import PatchError, Connection, SequenceEvent
    from signals.engine import EngineError
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the Signals directory and run with: uv run python tutorial_phase2.py")
    sys.exit(1)


class InteractiveTutorial:
    """Interactive tutorial for Phase 2 features."""
    
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
    
    def wait_for_enter(self, prompt: str = "Press Enter to continue..."):
        """Wait for user input."""
        input(f"\n💡 {prompt}")
    
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

    def tutorial_1_basic_concepts(self):
        """Tutorial 1: Basic Phase 2 concepts."""
        self.print_header("Tutorial 1: Understanding Phase 2 Concepts")
        
        print("""
🎯 Phase 2 introduces three major components:

1. 📋 Patch System - Load synthesizer configurations from YAML files
2. 🔗 Module Graph - Automatic signal routing and execution ordering  
3. 🎛️ SynthEngine - High-level API for synthesis control

Let's explore each component step by step!
        """)
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter("Ready to start with hands-on examples?")

    def tutorial_2_patch_basics(self):
        """Tutorial 2: Patch system basics."""
        self.print_header("Tutorial 2: Patch System Basics")
        
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
        
        self.create_demo_file("tutorial_simple.yaml", simple_patch, "simple patch file")
        
        print("""
📋 A patch file contains:
- 📝 Metadata (name, description, sample_rate)
- 🔧 Modules (synthesizer components with parameters)
- 🔗 Connections (how modules connect to each other)
- ⏰ Sequence (timed events for automation)
        """)
        
        self.wait_for_enter()
        
        self.print_step("Loading a Patch")
        
        self.show_code("""
# Create a Patch object from YAML file
from signals import Patch

patch = Patch.from_file("tutorial_simple.yaml")
print(f"Loaded: {patch.name}")
print(f"Modules: {list(patch.modules.keys())}")
print(f"Connections: {len(patch.connections)}")
        """)
        
        # Execute the code
        try:
            patch = Patch.from_file("tutorial_simple.yaml")
            print(f"\n✅ Loaded: {patch.name}")
            print(f"   Modules: {list(patch.modules.keys())}")
            print(f"   Connections: {len(patch.connections)}")
            print(f"   Sequence events: {len(patch.sequence)}")
            self.current_patch = patch
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter("Ready to learn about the SynthEngine?")

    def tutorial_3_synthengine_basics(self):
        """Tutorial 3: SynthEngine basics."""
        self.print_header("Tutorial 3: SynthEngine - The Control Center")
        
        self.print_step("Creating a SynthEngine")
        
        self.show_code("""
# SynthEngine is the main interface for synthesis
from signals import SynthEngine

engine = SynthEngine(sample_rate=48000)
print(f"Engine created with sample rate: {engine.sample_rate}")
        """)
        
        self.engine = SynthEngine(sample_rate=48000)
        print(f"\n✅ Engine created with sample rate: {self.engine.sample_rate}")
        
        self.wait_for_enter()
        
        self.print_step("Loading a Patch into the Engine")
        
        self.show_code("""
# Load our patch into the engine
patch = engine.load_patch("tutorial_simple.yaml")
print(f"Loaded patch: {patch.name}")

# Get patch information
info = engine.get_patch_info()
print(f"Modules: {info['modules']}")
print(f"Duration: {info['duration']}s")
        """)
        
        try:
            patch = self.engine.load_patch("tutorial_simple.yaml")
            print(f"\n✅ Loaded patch: {patch.name}")
            
            info = self.engine.get_patch_info()
            print(f"   Modules: {info['modules']}")
            print(f"   Duration: {info['duration']}s")
            print(f"   Sample rate: {info['sample_rate']} Hz")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter("Ready to explore dynamic parameter control?")

    def tutorial_4_dynamic_control(self):
        """Tutorial 4: Dynamic parameter control."""
        self.print_header("Tutorial 4: Dynamic Parameter Control")
        
        self.print_step("Examining Current Parameters")
        
        if self.engine and self.engine.current_patch:
            print("🔍 Let's see the current module parameters:")
            
            try:
                for module_id in self.engine.current_patch.modules.keys():
                    params = self.engine.get_module_parameters(module_id)
                    print(f"\n📦 {module_id}:")
                    for param, value in params.items():
                        print(f"   {param}: {value}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter()
        
        self.print_step("Testing the Changes")
        
        print("🎵 Let's render audio with the new parameters:")
        
        try:
            audio_data = self.engine.render(duration=0.5)
            features = self.engine.export_features(audio_data)
            print(f"✅ New audio: peak={features['peak']:.3f}, rms={features['rms']:.3f}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.wait_for_enter("Ready to learn about templates?")

    def tutorial_5_templates(self):
        """Tutorial 5: Template system."""
        self.print_header("Tutorial 5: Template System for Parameter Exploration")
        
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
        
        self.create_demo_file("tutorial_template.yaml", template_content, "template file")
        
        self.wait_for_enter()
        
        self.print_step("Creating a Template Object")
        
        self.show_code("""
# Create a template from the file
from signals import PatchTemplate

template = PatchTemplate("tutorial_template.yaml")
print(f"Template variables: {template.variables}")

# Get variable schema (default values)
schema = template.get_variable_schema()
print(f"Default values: {schema}")
        """)
        
        try:
            template = PatchTemplate("tutorial_template.yaml")
            print(f"\n✅ Template variables: {template.variables}")
            
            schema = template.get_variable_schema()
            print(f"   Default values: {schema}")
        except Exception as e:
            print(f"❌ Error: {e}")
            template = None
        
        self.wait_for_enter()
        
        self.print_step("Generating Variations")
        
        if template:
            print("🎵 Let's create different variations:")
            
            variations = [
                {"osc_freq": 220.0, "osc_wave": "sine", "env_attack": 0.01},
                {"osc_freq": 440.0, "osc_wave": "square", "env_attack": 0.05},
                {"osc_freq": 880.0, "osc_wave": "triangle", "env_attack": 0.1}
            ]
            
            for i, params in enumerate(variations, 1):
                print(f"\n🔸 Variation {i}: {params}")
                
                try:
                    # Load patch with parameters
                    patch = self.engine.load_patch("tutorial_template.yaml", params)
                    print(f"   ✅ Loaded: {patch.name}")
                    
                    # Quick render
                    audio = self.engine.render(duration=0.3)
                    features = self.engine.export_features(audio)
                    print(f"   📊 Peak: {features['peak']:.3f}, RMS: {features['rms']:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
        
        self.wait_for_enter("Ready to explore the module graph system?")

    def tutorial_6_module_graph(self):
        """Tutorial 6: Module graph system."""
        self.print_header("Tutorial 6: Module Graph - The Signal Routing Engine")
        
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
        
        self.create_demo_file("tutorial_complex.yaml", complex_patch, "complex patch file")
        
        self.wait_for_enter()
        
        self.print_step("Loading and Examining the Graph")
        
        try:
            patch = self.engine.load_patch("tutorial_complex.yaml")
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
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter()
        
        self.print_step("Rendering Complex Audio")
        
        try:
            print("🎵 Rendering the complex patch...")
            audio = self.engine.render(duration=1.5)
            features = self.engine.export_features(audio)
            
            print(f"✅ Complex audio generated:")
            print(f"   Length: {features['length_seconds']:.2f}s")
            print(f"   Peak: {features['peak']:.3f}")
            print(f"   RMS: {features['rms']:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.wait_for_enter("Ready for the advanced features?")

    def tutorial_7_advanced_features(self):
        """Tutorial 7: Advanced features."""
        self.print_header("Tutorial 7: Advanced Features for External Control")
        
        self.print_step("Batch Processing")
        
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
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter()
        
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
        
        self.wait_for_enter("Ready for the summary?")

    def tutorial_8_summary(self):
        """Tutorial 8: Summary and next steps."""
        self.print_header("Tutorial 8: Summary and Next Steps")
        
        print("""
🎉 Congratulations! You've explored all major Phase 2 features:

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
  - Cycle detection and prevention

✅ External Integration
  - Programmatic Python API
  - Batch processing capabilities
  - Ready for parameter exploration
        """)
        
        self.wait_for_enter()
        
        print("""
🚀 What you can do now:

1. Create your own patch files
2. Build parameterized templates
3. Use the SynthEngine in your projects
4. Integrate with external programs
5. Explore parameter spaces systematically

📁 Files created during this tutorial:
        """)
        
        for filepath in self.created_files:
            print(f"   - {filepath}")
        
        print("""
💡 Try these next steps:

1. Modify the tutorial patches
2. Create your own templates
3. Experiment with different module types
4. Build complex module graphs
5. Integrate with your own Python scripts

Happy synthesizing! 🎵
        """)

    def run_tutorial(self):
        """Run the complete interactive tutorial."""
        print("""
🎵 Welcome to the Signals Phase 2 Interactive Tutorial!

This tutorial will guide you through all the new features step by step.
You'll learn by doing - each concept is demonstrated with working code.
        """)
        
        self.wait_for_enter("Ready to start? (Ctrl+C to exit anytime)")
        
        try:
            self.tutorial_1_basic_concepts()
            self.tutorial_2_patch_basics()
            self.tutorial_3_synthengine_basics()
            self.tutorial_4_dynamic_control()
            self.tutorial_5_templates()
            self.tutorial_6_module_graph()
            self.tutorial_7_advanced_features()
            self.tutorial_8_summary()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Tutorial interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Tutorial error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up...")
            self.cleanup()
            print("\nThanks for trying the Phase 2 tutorial! 👋")


def main():
    """Main entry point."""
    tutorial = InteractiveTutorial()
    tutorial.run_tutorial()


if __name__ == "__main__":
    main()