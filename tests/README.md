# Signals Test System

## Overview

This directory contains a comprehensive test suite for the Signals synthesizer framework. The test system is designed to ensure code quality, reliability, and maintainability throughout the development lifecycle.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── test_basic.py            # Legacy basic tests (Phase 1)
├── test_phase2.py           # Legacy Phase 2 tests
├── test_modules.py          # Comprehensive module tests
├── test_engine.py           # Engine and integration tests
└── README.md               # This file
```

## Test Categories

The test suite is organized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for component interactions  
- `@pytest.mark.performance` - Performance and benchmark tests
- `@pytest.mark.audio` - Tests that generate or process audio
- `@pytest.mark.patch` - Tests for patch loading and processing
- `@pytest.mark.logging` - Tests for logging functionality
- `@pytest.mark.slow` - Tests that take longer to run

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_modules.py

# Run specific test class
python -m pytest tests/test_modules.py::TestOscillator

# Run specific test method
python -m pytest tests/test_modules.py::TestOscillator::test_oscillator_initialization
```

### Category-Based Testing

```bash
# Run only unit tests
python -m pytest tests/ -m unit

# Run only integration tests
python -m pytest tests/ -m integration

# Run performance tests
python -m pytest tests/ -m performance

# Run audio processing tests
python -m pytest tests/ -m audio

# Exclude slow tests
python -m pytest tests/ -m "not slow"
```

### Verbose Output and Coverage

```bash
# Verbose output
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src/signals --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/ --cov=src/signals --cov-report=html
```

## Test Fixtures

The `conftest.py` file provides shared fixtures for all tests:

### Module Fixtures
- `oscillator_module` - Create oscillator instances
- `envelope_module` - Create envelope instances
- `vca_module` - Create VCA instances
- `mixer_module` - Create mixer instances
- `synth_engine` - Create synthesis engine instances

### Data Fixtures
- `sample_rates` - Standard sample rates for testing
- `test_frequencies` - Standard test frequencies
- `waveform_types` - Available waveform types
- `basic_patch_data` - Basic patch configuration
- `complex_patch_data` - Complex patch configuration
- `template_patch_content` - Template patch content

### Utility Fixtures
- `temp_dir` - Temporary directory for test files
- `sample_audio_data` - Generate sample audio
- `audio_validator` - Audio validation utilities
- `patch_validator` - Patch validation utilities
- `performance_timer` - Performance timing utility
- `signal_generators` - Signal generation functions

## Test Coverage Areas

### Module Tests (`test_modules.py`)

#### Oscillator Tests
- Initialization with different sample rates
- All waveform types (sine, square, saw, triangle, noise)
- Parameter changes (frequency, amplitude, waveform)
- Phase continuity across parameter changes
- Invalid parameter handling

#### Envelope Tests
- ADSR phase progression
- Parameter formats (seconds, percentages, auto)
- Trigger signal handling
- Relative parameter calculations
- Edge cases (zero times, extreme values)

#### VCA Tests
- Basic amplitude modulation
- Gain parameter control
- Signal type combinations
- Missing input handling
- Extreme value processing

#### Mixer Tests
- Multi-channel mixing
- Individual gain controls
- Signal type filtering
- Partial input handling
- Negative gain values

#### Output Tests
- WAV file generation
- Different bit depths
- Audio signal processing
- Non-audio signal filtering
- Silence addition

### Engine Tests (`test_engine.py`)

#### SynthEngine Tests
- Engine initialization
- Patch loading from files and dictionaries
- Audio rendering with various durations
- Automatic duration calculation
- Dynamic parameter setting
- Feature extraction
- Error handling

#### Patch System Tests
- Patch validation
- Connection verification
- Sequence validation
- Dictionary serialization roundtrip
- Duration calculation

#### Template System Tests
- Variable extraction
- Schema generation
- Template instantiation
- Default value handling
- Error handling

#### Integration Tests
- Complete synthesis pipeline
- Template batch processing
- Dynamic parameter changes
- Performance characteristics
- Error recovery
- Memory cleanup

## Writing New Tests

### Test Class Structure

```python
@pytest.mark.unit
class TestNewModule:
    """Tests for NewModule."""
    
    def test_initialization(self, sample_rates):
        """Test module initialization."""
        for sample_rate in sample_rates:
            module = NewModule(sample_rate)
            assert module.sample_rate == sample_rate
    
    def test_parameter_setting(self, new_module_fixture):
        """Test parameter setting."""
        module = new_module_fixture()
        module.set_parameter("param", value)
        assert module.param == value
    
    def test_processing(self, new_module_fixture):
        """Test signal processing."""
        module = new_module_fixture()
        input_signal = Signal(SignalType.AUDIO, 0.5)
        output = module.process([input_signal])
        assert len(output) == 1
        assert output[0].type == SignalType.AUDIO
```

### Fixture Creation

```python
@pytest.fixture
def new_module_fixture():
    """Create NewModule for testing."""
    def _create(sample_rate=48000, **params):
        module = NewModule(sample_rate)
        for param, value in params.items():
            module.set_parameter(param, value)
        return module
    return _create
```

### Audio Validation

```python
def test_audio_output(self, audio_validator):
    """Test audio output quality."""
    audio_data = generate_audio()
    
    assert audio_validator.is_valid_range(audio_data)
    assert audio_validator.is_not_silent(audio_data)
    assert audio_validator.has_no_clipping(audio_data)
    assert audio_validator.has_expected_length(
        audio_data, sample_rate, duration
    )
```

## Best Practices

### Test Design
1. **Single Responsibility** - Each test should verify one specific behavior
2. **Descriptive Names** - Test names should clearly describe what is being tested
3. **Arrange-Act-Assert** - Structure tests with clear setup, execution, and verification
4. **Independent Tests** - Tests should not depend on each other
5. **Deterministic** - Tests should produce consistent results

### Performance Testing
1. Use `@pytest.mark.performance` for performance tests
2. Include reasonable performance bounds
3. Test scaling characteristics
4. Monitor memory usage for long-running tests

### Audio Testing
1. Validate audio range (-1.0 to 1.0)
2. Check for silence when appropriate
3. Verify expected duration and sample count
4. Test for clipping and artifacts

### Error Testing
1. Test both valid and invalid inputs
2. Verify proper exception types
3. Test error recovery
4. Ensure graceful degradation

## Continuous Integration

The test suite is designed to run in CI environments:

```bash
# Quick test suite (unit tests only)
python -m pytest tests/ -m unit --tb=short

# Full test suite with coverage
python -m pytest tests/ --cov=src/signals --cov-report=xml

# Performance regression tests
python -m pytest tests/ -m performance --tb=short
```

## Troubleshooting

### Common Issues

1. **Import Errors** - Ensure `src/signals` is in Python path
2. **Fixture Not Found** - Check `conftest.py` for fixture definitions
3. **Audio Tests Failing** - Verify audio validation criteria
4. **Slow Tests** - Use `pytest.mark.slow` and run selectively

### Debug Mode

```bash
# Run with verbose output and stop on first failure
python -m pytest tests/ -v -x

# Run with pdb on failures
python -m pytest tests/ --pdb

# Run specific test with full traceback
python -m pytest tests/test_modules.py::TestClass::test_method -vvs
```

## Contributing

When adding new features:

1. Write tests before implementation (TDD)
2. Ensure comprehensive coverage of new code
3. Add appropriate test markers
4. Update fixtures if needed
5. Document complex test scenarios
6. Run full test suite before submitting

## Test Data

Test data should be:
- Deterministic and reproducible
- Minimal but representative
- Easy to understand and modify
- Isolated from external dependencies

The test system provides comprehensive validation of the Signals synthesizer framework, ensuring reliability and maintainability throughout the development process.