#!/usr/bin/env python3
"""
Logging system demonstration script for the Signals synthesizer framework.

This script demonstrates various logging capabilities including:
- Different log levels
- Module-specific logging
- Performance logging
- File output
- Dynamic log level changes
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import SynthEngine, configure_logging, set_module_log_level, LogLevel
from signals.core.logging import LogContext, enable_performance_logging
import time


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("=== Basic Logging Demo ===")
    
    # Configure logging
    configure_logging(level=LogLevel.INFO, console=True)
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    print("\n✅ Basic logging configured and patch loaded")


def demo_log_levels():
    """Demonstrate different log levels."""
    print("\n=== Log Levels Demo ===")
    
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    
    for level in levels:
        print(f"\n--- Setting log level to {level} ---")
        configure_logging(level=level, console=True)
        
        engine = SynthEngine(sample_rate=48000)
        # This will show different amounts of detail based on log level
        
        # Force some parameter changes to generate logs
        engine.load_patch("examples/patches/basic_synth.yaml")
        
        time.sleep(0.5)  # Brief pause between demos


def demo_module_specific_logging():
    """Demonstrate module-specific log level control."""
    print("\n=== Module-Specific Logging Demo ===")
    
    # Set overall level to WARNING
    configure_logging(level=LogLevel.WARNING, console=True)
    
    # But enable DEBUG for specific modules
    print("\n--- Enabling DEBUG for envelope module only ---")
    set_module_log_level('envelope', 'DEBUG')
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    # Change envelope parameters to see debug logs
    engine.set_module_parameter('env1', 'attack', 0.1)
    engine.set_module_parameter('env1', 'release', 'auto')
    
    print("\n--- Now enabling DEBUG for oscillator module ---")
    set_module_log_level('oscillator', 'DEBUG')
    
    # Change oscillator parameters
    engine.set_module_parameter('osc1', 'frequency', 880.0)
    engine.set_module_parameter('osc1', 'waveform', 'square')


def demo_performance_logging():
    """Demonstrate performance logging."""
    print("\n=== Performance Logging Demo ===")
    
    configure_logging(level=LogLevel.DEBUG, console=True, performance_logging=True)
    enable_performance_logging(True)
    
    print("\n--- Performance logging enabled - rendering short audio ---")
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    # Render a short audio to see performance logs
    audio = engine.render(duration=0.1)  # Just 0.1 seconds
    
    print(f"\n✅ Rendered {len(audio)} samples with performance logging")


def demo_log_context():
    """Demonstrate temporary log level changes with context manager."""
    print("\n=== Log Context Demo ===")
    
    configure_logging(level=LogLevel.ERROR, console=True)
    
    engine = SynthEngine(sample_rate=48000)
    
    print("\n--- Normal operation (ERROR level) ---")
    engine.load_patch("examples/patches/basic_synth.yaml")
    
    print("\n--- Temporarily enable DEBUG for envelope module ---")
    with LogContext('modules.envelope', 'DEBUG'):
        # Inside this context, envelope module logs at DEBUG level
        engine.set_module_parameter('env1', 'attack', '5%')
        engine.set_module_parameter('env1', 'decay', 'auto')
    
    print("\n--- Back to ERROR level ---")
    engine.set_module_parameter('env1', 'sustain', 0.8)


def demo_file_logging():
    """Demonstrate file logging."""
    print("\n=== File Logging Demo ===")
    
    log_file = "demo_logging_output.log"
    
    configure_logging(
        level=LogLevel.INFO,
        console=True,
        file_path=log_file
    )
    
    print(f"\n--- Logging to both console and file: {log_file} ---")
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    audio = engine.render(duration=0.5)
    
    print(f"\n✅ Check {log_file} for file output")
    
    # Show first few lines of the log file
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[:5]
        print("\nFirst 5 lines of log file:")
        for line in lines:
            print(f"  {line.rstrip()}")
    except FileNotFoundError:
        print("Log file not created yet")


def demo_error_logging():
    """Demonstrate error logging."""
    print("\n=== Error Logging Demo ===")
    
    configure_logging(level=LogLevel.WARNING, console=True)
    
    engine = SynthEngine(sample_rate=48000)
    
    print("\n--- Causing intentional errors to see error logs ---")
    
    try:
        # Try to load non-existent patch
        engine.load_patch("non_existent_patch.yaml")
    except Exception:
        pass
    
    try:
        # Try to set invalid parameter
        patch = engine.load_patch("examples/patches/basic_synth.yaml")
        engine.set_module_parameter('osc1', 'invalid_parameter', 123)
    except Exception:
        pass
    
    try:
        # Try to set parameter on non-existent module
        engine.set_module_parameter('non_existent_module', 'frequency', 440)
    except Exception:
        pass


def main():
    """Run all logging demonstrations."""
    print("🎵 Signals Logging System Demonstration")
    print("=" * 50)
    
    try:
        demo_basic_logging()
        demo_log_levels()
        demo_module_specific_logging()
        demo_performance_logging()
        demo_log_context()
        demo_file_logging()
        demo_error_logging()
        
        print("\n" + "=" * 50)
        print("✅ All logging demonstrations completed successfully!")
        print("\nLogging system features demonstrated:")
        print("  • Basic logging configuration")
        print("  • Multiple log levels (DEBUG, INFO, WARNING, ERROR)")
        print("  • Module-specific log level control")
        print("  • Performance logging for audio processing")
        print("  • Context managers for temporary log level changes")
        print("  • File output with rotation")
        print("  • Structured error logging")
        print("\nFor production use:")
        print("  • Use WARNING or ERROR level for normal operation")
        print("  • Enable DEBUG/INFO for troubleshooting specific issues")
        print("  • Use performance logging for optimization work")
        print("  • Configure file logging for long-running processes")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())