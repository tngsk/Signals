#!/usr/bin/env python3
"""
Logging Usage Examples for the Signals Synthesizer Framework

This file contains practical examples of how to use the logging system
for different scenarios in audio synthesis and debugging.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals import SynthEngine, configure_logging, set_module_log_level, LogLevel
from signals.core.logging import LogContext, enable_performance_logging, get_logger


def example_1_basic_usage():
    """Example 1: Basic logging setup for normal operation"""
    print("=== Example 1: Basic Usage ===")
    
    # Configure logging for normal operation
    configure_logging(level=LogLevel.INFO, console=True)
    
    # Create engine and process audio
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    audio = engine.render(duration=1.0)
    
    print(f"Generated {len(audio)} samples\n")


def example_2_debugging_specific_module():
    """Example 2: Debug specific module while keeping others quiet"""
    print("=== Example 2: Module-Specific Debugging ===")
    
    # Keep most logging quiet
    configure_logging(level=LogLevel.WARNING, console=True)
    
    # But enable detailed logging for envelope module
    set_module_log_level('envelope', 'DEBUG')
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    # Modify envelope parameters to see debug output
    engine.set_module_parameter('env1', 'attack', '10%')
    engine.set_module_parameter('env1', 'release', 'auto')
    
    print("Envelope debugging completed\n")


def example_3_performance_analysis():
    """Example 3: Performance analysis for optimization"""
    print("=== Example 3: Performance Analysis ===")
    
    # Enable performance logging
    configure_logging(
        level=LogLevel.DEBUG, 
        console=True,
        performance_logging=True
    )
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/multi_osc_synth.yaml")
    
    print("Rendering audio with performance monitoring...")
    audio = engine.render(duration=0.5)  # Short duration for demo
    
    print(f"Performance analysis completed for {len(audio)} samples\n")


def example_4_file_logging():
    """Example 4: Log to file for batch processing or long-running tasks"""
    print("=== Example 4: File Logging ===")
    
    # Configure file logging
    configure_logging(
        level=LogLevel.INFO,
        console=True,
        file_path="signals_batch_processing.log"
    )
    
    logger = get_logger('batch_processor')
    logger.info("Starting batch processing session")
    
    engine = SynthEngine(sample_rate=48000)
    
    # Simulate batch processing
    patches = ["examples/patches/basic_synth.yaml"]
    
    for i, patch_file in enumerate(patches):
        logger.info(f"Processing patch {i+1}/{len(patches)}: {patch_file}")
        
        patch = engine.load_patch(patch_file)
        audio = engine.render(duration=1.0)
        
        logger.info(f"Completed patch {i+1}: {len(audio)} samples generated")
    
    logger.info("Batch processing session completed")
    print("Check signals_batch_processing.log for detailed logs\n")


def example_5_temporary_debug_context():
    """Example 5: Temporarily increase logging for specific operations"""
    print("=== Example 5: Temporary Debug Context ===")
    
    # Normal operation level
    configure_logging(level=LogLevel.ERROR, console=True)
    
    engine = SynthEngine(sample_rate=48000)
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    print("Normal operation (minimal logging)...")
    engine.set_module_parameter('osc1', 'frequency', 440.0)
    
    print("Temporarily enabling debug for VCA module...")
    with LogContext('modules.vca', 'DEBUG'):
        # Inside this context, VCA logs at DEBUG level
        engine.set_module_parameter('vca1', 'gain', 0.8)
        engine.set_module_parameter('vca1', 'gain', 1.2)
    
    print("Back to minimal logging...")
    engine.set_module_parameter('osc1', 'amplitude', 0.9)
    
    print("Temporary debug context example completed\n")


def example_6_error_tracking():
    """Example 6: Comprehensive error tracking and debugging"""
    print("=== Example 6: Error Tracking ===")
    
    # Configure for error tracking
    configure_logging(
        level=LogLevel.WARNING,
        console=True,
        file_path="error_tracking.log"
    )
    
    logger = get_logger('error_tracker')
    
    engine = SynthEngine(sample_rate=48000)
    
    # Simulate various error conditions
    test_cases = [
        ("valid_patch", "examples/patches/basic_synth.yaml"),
        ("invalid_patch", "non_existent.yaml"),
        ("invalid_parameter", None)  # Special case
    ]
    
    for test_name, patch_file in test_cases:
        logger.info(f"Testing: {test_name}")
        
        try:
            if patch_file:
                patch = engine.load_patch(patch_file)
                logger.info(f"Successfully loaded: {patch_file}")
            else:
                # Test invalid parameter
                patch = engine.load_patch("examples/patches/basic_synth.yaml")
                engine.set_module_parameter('osc1', 'invalid_param', 123)
                
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
    
    print("Error tracking example completed - check error_tracking.log\n")


def example_7_production_configuration():
    """Example 7: Recommended production configuration"""
    print("=== Example 7: Production Configuration ===")
    
    # Production-ready logging configuration
    configure_logging(
        level=LogLevel.WARNING,  # Only warnings and errors
        console=False,           # No console output in production
        file_path="signals_production.log"
    )
    
    logger = get_logger('production')
    logger.info("Starting production audio processing")
    
    try:
        engine = SynthEngine(sample_rate=48000)
        patch = engine.load_patch("examples/patches/basic_synth.yaml")
        audio = engine.render(duration=2.0)
        
        logger.info(f"Successfully generated {len(audio)} samples")
        
    except Exception as e:
        logger.error(f"Production processing failed: {e}")
        raise
    
    print("Production configuration example completed\n")


def example_8_development_workflow():
    """Example 8: Typical development workflow with adaptive logging"""
    print("=== Example 8: Development Workflow ===")
    
    # Start with INFO level for development
    configure_logging(level=LogLevel.INFO, console=True)
    
    logger = get_logger('development')
    logger.info("Starting development session")
    
    engine = SynthEngine(sample_rate=48000)
    
    # Phase 1: Load and test basic functionality
    logger.info("Phase 1: Basic functionality test")
    patch = engine.load_patch("examples/patches/basic_synth.yaml")
    
    # Phase 2: Debug specific issues
    logger.info("Phase 2: Debugging envelope behavior")
    set_module_log_level('envelope', 'DEBUG')
    
    engine.set_module_parameter('env1', 'attack', '5%')
    engine.set_module_parameter('env1', 'decay', 'auto')
    
    # Phase 3: Performance optimization
    logger.info("Phase 3: Performance optimization")
    enable_performance_logging(True)
    
    audio = engine.render(duration=0.1)  # Short test
    
    # Phase 4: Clean up for final testing
    logger.info("Phase 4: Final testing with minimal logging")
    configure_logging(level=LogLevel.WARNING, console=True)
    
    final_audio = engine.render(duration=1.0)
    
    print(f"Development workflow completed: {len(final_audio)} samples\n")


def main():
    """Run all logging examples"""
    print("🎵 Signals Logging System Examples")
    print("=" * 50)
    
    examples = [
        example_1_basic_usage,
        example_2_debugging_specific_module,
        example_3_performance_analysis,
        example_4_file_logging,
        example_5_temporary_debug_context,
        example_6_error_tracking,
        example_7_production_configuration,
        example_8_development_workflow
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}\n")
    
    print("=" * 50)
    print("📋 Logging Best Practices Summary:")
    print()
    print("🔧 Development:")
    print("  • Use INFO or DEBUG level for active development")
    print("  • Enable module-specific debugging for targeted issues")
    print("  • Use performance logging for optimization work")
    print()
    print("🚀 Production:")
    print("  • Use WARNING or ERROR level only")
    print("  • Always log to files, not console")
    print("  • Include error tracking and monitoring")
    print()
    print("🛠️ Debugging:")
    print("  • Use LogContext for temporary debug sessions")
    print("  • Enable specific modules rather than global DEBUG")
    print("  • Combine file logging with module-specific levels")
    print()
    print("⚡ Performance:")
    print("  • Only enable performance logging during optimization")
    print("  • Use INFO level for normal operation monitoring")
    print("  • Consider log rotation for long-running processes")


if __name__ == "__main__":
    main()