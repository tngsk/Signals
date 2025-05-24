"""
Tests for context-based sample rate management.

This module tests the SynthContext system that allows modules to be created
without explicit sample rate specification while maintaining consistency.
"""

import pytest
import threading
import time
from unittest.mock import patch

from signals import (
    SynthContext, synthesis_context, get_sample_rate_or_default, ContextError,
    Oscillator, EnvelopeADSR, VCA, SynthEngine
)


@pytest.mark.unit
class TestSynthContext:
    """Test SynthContext functionality."""
    
    def test_context_creation(self):
        """Test basic context creation and properties."""
        context = SynthContext(sample_rate=48000, buffer_size=1024)
        assert context.sample_rate == 48000
        assert context.parameters == {'buffer_size': 1024}
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with SynthContext(sample_rate=44100) as ctx:
            assert ctx.sample_rate == 44100
            assert SynthContext.has_context()
            assert SynthContext.get_sample_rate() == 44100
        
        # Context should be cleared after exit
        assert not SynthContext.has_context()
    
    def test_nested_contexts(self):
        """Test nested context management."""
        with SynthContext(sample_rate=48000):
            assert SynthContext.get_sample_rate() == 48000
            
            with SynthContext(sample_rate=44100):
                assert SynthContext.get_sample_rate() == 44100
            
            # Should restore previous context
            assert SynthContext.get_sample_rate() == 48000
    
    def test_context_without_manager(self):
        """Test that accessing context without manager raises error."""
        with pytest.raises(ContextError, match="No synthesis context available"):
            SynthContext.get_sample_rate()
    
    def test_get_parameter(self):
        """Test parameter retrieval from context."""
        with SynthContext(sample_rate=48000, test_param="test_value"):
            assert SynthContext.get_parameter("test_param") == "test_value"
            assert SynthContext.get_parameter("nonexistent", "default") == "default"
    
    def test_context_info(self):
        """Test context information retrieval."""
        with SynthContext(sample_rate=48000, custom_param=123):
            info = SynthContext.get_context_info()
            assert info['sample_rate'] == 48000
            assert info['parameters']['custom_param'] == 123
            assert 'thread_id' in info
            assert info['context_depth'] == 0
    
    def test_temporary_context(self):
        """Test temporary context creation."""
        with SynthContext.temporary(96000, temp_param="temp") as ctx:
            assert ctx.sample_rate == 96000
            assert SynthContext.get_parameter("temp_param") == "temp"


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions for context management."""
    
    def test_get_sample_rate_or_default_with_explicit(self):
        """Test utility function with explicit sample rate."""
        result = get_sample_rate_or_default(44100)
        assert result == 44100
    
    def test_get_sample_rate_or_default_with_context(self):
        """Test utility function with context sample rate."""
        with SynthContext(sample_rate=96000):
            result = get_sample_rate_or_default()
            assert result == 96000
    
    def test_get_sample_rate_or_default_with_default(self):
        """Test utility function with default value."""
        result = get_sample_rate_or_default(default=22050)
        assert result == 22050
    
    def test_synthesis_context_convenience(self):
        """Test synthesis_context convenience function."""
        with synthesis_context(48000, test_param="test") as ctx:
            assert ctx.sample_rate == 48000
            assert SynthContext.get_parameter("test_param") == "test"


@pytest.mark.unit
class TestModuleContextIntegration:
    """Test module integration with context system."""
    
    def test_oscillator_with_context(self):
        """Test Oscillator creation with context."""
        with SynthContext(sample_rate=44100):
            osc = Oscillator()
            assert osc.sample_rate == 44100
    
    def test_oscillator_explicit_overrides_context(self):
        """Test that explicit sample rate overrides context."""
        with SynthContext(sample_rate=44100):
            osc = Oscillator(sample_rate=48000)
            assert osc.sample_rate == 48000
    
    def test_oscillator_without_context_uses_default(self):
        """Test Oscillator without context uses default."""
        osc = Oscillator()
        assert osc.sample_rate == 48000  # Default value
    
    def test_envelope_with_context(self):
        """Test EnvelopeADSR creation with context."""
        with SynthContext(sample_rate=96000):
            env = EnvelopeADSR()
            assert env.sample_rate == 96000
    
    def test_vca_with_context(self):
        """Test VCA creation with context."""
        with SynthContext(sample_rate=22050):
            vca = VCA()
            assert vca.sample_rate == 22050
    
    def test_multiple_modules_same_context(self):
        """Test multiple modules in same context."""
        with SynthContext(sample_rate=48000):
            osc = Oscillator()
            env = EnvelopeADSR()
            vca = VCA()
            
            assert osc.sample_rate == 48000
            assert env.sample_rate == 48000
            assert vca.sample_rate == 48000


@pytest.mark.integration
class TestSynthEngineContextIntegration:
    """Test SynthEngine context integration."""
    
    def test_engine_as_context_manager(self):
        """Test SynthEngine as context manager."""
        with SynthEngine(sample_rate=44100) as engine:
            assert SynthContext.has_context()
            assert SynthContext.get_sample_rate() == 44100
            
            # Create modules within engine context
            osc = Oscillator()
            env = EnvelopeADSR()
            
            assert osc.sample_rate == 44100
            assert env.sample_rate == 44100
    
    def test_engine_context_method(self):
        """Test SynthEngine.context() method."""
        engine = SynthEngine(sample_rate=96000)
        
        with engine.context():
            assert SynthContext.get_sample_rate() == 96000
            
            osc = Oscillator()
            assert osc.sample_rate == 96000
    
    def test_engine_patch_loading_with_context(self, temp_dir, basic_patch_data, create_patch_file):
        """Test patch loading maintains context."""
        patch_file = create_patch_file(temp_dir, basic_patch_data)
        
        with SynthEngine(sample_rate=44100) as engine:
            patch = engine.load_patch(patch_file)
            assert patch.sample_rate == 44100


@pytest.mark.unit
class TestThreadSafety:
    """Test thread safety of context system."""
    
    def test_thread_isolation(self):
        """Test that contexts are isolated between threads."""
        results = {}
        
        def worker(thread_id, sample_rate):
            with SynthContext(sample_rate=sample_rate):
                time.sleep(0.1)  # Allow context switching
                results[thread_id] = SynthContext.get_sample_rate()
        
        # Start multiple threads with different sample rates
        threads = []
        for i, rate in enumerate([44100, 48000, 96000]):
            thread = threading.Thread(target=worker, args=(i, rate))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify each thread maintained its own context
        assert results[0] == 44100
        assert results[1] == 48000
        assert results[2] == 96000
    
    def test_no_context_leakage_between_threads(self):
        """Test that context doesn't leak between threads."""
        def worker_with_context():
            with SynthContext(sample_rate=44100):
                pass  # Context should be cleaned up
        
        def worker_without_context():
            # This should not have access to the other thread's context
            assert not SynthContext.has_context()
        
        thread1 = threading.Thread(target=worker_with_context)
        thread2 = threading.Thread(target=worker_without_context)
        
        thread1.start()
        thread1.join()
        
        thread2.start()
        thread2.join()


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in context system."""
    
    def test_context_error_when_no_context(self):
        """Test ContextError when no context is available."""
        with pytest.raises(ContextError):
            SynthContext.get_current()
    
    def test_context_error_message(self):
        """Test ContextError provides helpful message."""
        with pytest.raises(ContextError, match="No synthesis context available"):
            SynthContext.get_sample_rate()
    
    def test_exception_cleanup(self):
        """Test context is cleaned up even when exception occurs."""
        with pytest.raises(ValueError):
            with SynthContext(sample_rate=48000):
                assert SynthContext.has_context()
                raise ValueError("Test exception")
        
        # Context should be cleaned up despite exception
        assert not SynthContext.has_context()


@pytest.mark.performance
class TestContextPerformance:
    """Test performance characteristics of context system."""
    
    def test_context_creation_performance(self):
        """Test that context creation is fast."""
        import time
        
        start_time = time.perf_counter()
        for _ in range(1000):
            with SynthContext(sample_rate=48000):
                pass
        end_time = time.perf_counter()
        
        # Should complete 1000 context creations in under 100ms
        total_time = end_time - start_time
        assert total_time < 0.1, f"Context creation too slow: {total_time:.3f}s"
    
    def test_sample_rate_access_performance(self):
        """Test that sample rate access is fast."""
        import time
        
        with SynthContext(sample_rate=48000):
            start_time = time.perf_counter()
            for _ in range(10000):
                rate = SynthContext.get_sample_rate()
                assert rate == 48000
            end_time = time.perf_counter()
        
        # Should complete 10000 accesses in under 10ms
        total_time = end_time - start_time
        assert total_time < 0.01, f"Sample rate access too slow: {total_time:.3f}s"


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with explicit sample rates."""
    
    def test_explicit_sample_rate_still_works(self):
        """Test that modules still work with explicit sample rates."""
        osc = Oscillator(sample_rate=44100)
        env = EnvelopeADSR(sample_rate=44100)
        vca = VCA(sample_rate=44100)
        
        assert osc.sample_rate == 44100
        assert env.sample_rate == 44100
        assert vca.sample_rate == 44100
    
    def test_mixed_explicit_and_context(self):
        """Test mixing explicit and context-based creation."""
        with SynthContext(sample_rate=48000):
            osc_context = Oscillator()  # Uses context
            osc_explicit = Oscillator(sample_rate=44100)  # Explicit override
            
            assert osc_context.sample_rate == 48000
            assert osc_explicit.sample_rate == 44100
    
    def test_existing_code_patterns_unchanged(self):
        """Test that existing code patterns continue to work."""
        # Old pattern: explicit sample rates
        sample_rate = 48000
        osc = Oscillator(sample_rate=sample_rate)
        env = EnvelopeADSR(sample_rate=sample_rate)
        
        assert osc.sample_rate == 48000
        assert env.sample_rate == 48000
        
        # Should work exactly the same as before


@pytest.mark.integration
class TestRealWorldUsage:
    """Test real-world usage patterns."""
    
    def test_patch_creation_workflow(self):
        """Test typical patch creation workflow."""
        with SynthEngine(sample_rate=48000) as engine:
            # Create modules without specifying sample rate
            osc1 = Oscillator()
            osc2 = Oscillator()
            env = EnvelopeADSR()
            vca = VCA()
            
            # All should use engine's sample rate
            assert osc1.sample_rate == 48000
            assert osc2.sample_rate == 48000
            assert env.sample_rate == 48000
            assert vca.sample_rate == 48000
    
    def test_different_sample_rates_workflow(self):
        """Test workflow with different sample rates."""
        # High quality rendering
        with synthesis_context(96000):
            osc_hq = Oscillator()
            assert osc_hq.sample_rate == 96000
        
        # Standard quality rendering
        with synthesis_context(48000):
            osc_std = Oscillator()
            assert osc_std.sample_rate == 48000
        
        # Low quality/demo rendering
        with synthesis_context(22050):
            osc_demo = Oscillator()
            assert osc_demo.sample_rate == 22050
    
    def test_module_factory_pattern(self):
        """Test module factory pattern with context."""
        def create_basic_synth():
            """Factory function that creates a basic synth setup."""
            # Assumes it's called within a synthesis context
            osc = Oscillator()
            env = EnvelopeADSR()
            vca = VCA()
            return osc, env, vca
        
        with SynthContext(sample_rate=44100):
            osc, env, vca = create_basic_synth()
            
            assert osc.sample_rate == 44100
            assert env.sample_rate == 44100
            assert vca.sample_rate == 44100