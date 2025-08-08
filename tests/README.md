# DevGuard Testing Infrastructure

This directory contains the comprehensive testing infrastructure for DevGuard, including unit tests, integration tests, performance tests, security tests, and all necessary fixtures and utilities.

## 📁 Directory Structure

```
tests/
├── README.md                 # This file
├── conftest.py              # Main pytest configuration and fixtures
├── pytest.ini              # Pytest configuration
├── unit/                    # Unit tests
│   ├── __init__.py
│   └── test_sample.py       # Sample tests demonstrating infrastructure
├── integration/             # Integration tests
│   └── __init__.py
├── performance/             # Performance and benchmark tests
│   └── __init__.py
├── security/                # Security tests
│   └── __init__.py
├── utils/                   # Test utilities and helpers
│   ├── __init__.py
│   ├── assertions.py        # Custom assertion helpers
│   └── helpers.py           # Test helper functions
└── data/                    # Test data files
    └── sample_code.py       # Sample code for testing
```

## 🚀 Quick Start

### Setup Development Environment

```bash
# Run the setup script
./scripts/setup-dev.sh

# Or manually:
pip install -e ".[dev]"
pre-commit install
```

### Run Tests

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-performance
make test-security

# Run tests with coverage
make coverage

# Run quality checks
make quality
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests that test individual components without external dependencies.

**Characteristics:**
- Run in < 1 second each
- Use mocks for external dependencies
- Test single functions/classes
- High coverage requirements (95%+)

**Example:**
```python
@pytest.mark.unit
def test_memory_entry_creation(mock_shared_memory):
    entry = create_memory_entry("test_agent", "observation", {"data": "test"})
    assert entry.agent_id == "test_agent"
    assert entry.type == "observation"
```

### Integration Tests (`tests/integration/`)

Tests that verify component interactions and end-to-end workflows.

**Characteristics:**
- May use real databases/services
- Test component integration
- Longer execution time allowed
- Focus on workflow validation

**Example:**
```python
@pytest.mark.integration
async def test_agent_coordination(mock_multi_repos, mock_vector_db):
    # Test full agent coordination workflow
    swarm = DevGuardSwarm(config)
    result = await swarm.process_repository_changes()
    assert result.success
```

### Performance Tests (`tests/performance/`)

Benchmark tests that measure performance and detect regressions.

**Characteristics:**
- Use pytest-benchmark
- Measure execution time and memory
- Set performance thresholds
- Generate benchmark reports

**Example:**
```python
@pytest.mark.performance
def test_vector_search_performance(benchmark, large_dataset):
    result = benchmark(vector_db.search, "test query")
    assert len(result) > 0
```

### Security Tests (`tests/security/`)

Tests that validate security measures and detect vulnerabilities.

**Characteristics:**
- Test input validation
- Check for common vulnerabilities
- Validate security policies
- Test authentication/authorization

**Example:**
```python
@pytest.mark.security
def test_sql_injection_prevention():
    with pytest.raises(SecurityError):
        unsafe_query(malicious_input)
```

## 🔧 Test Fixtures

### Core Fixtures

#### Configuration Fixtures
- `test_config`: Basic test configuration
- `temp_dir`: Temporary directory for test files
- `mock_file_system`: Mock file system structure

#### Repository Fixtures
- `mock_git_repo`: Single mock Git repository
- `mock_multi_repos`: Multiple mock repositories for cross-repo testing

#### LLM Fixtures
- `mock_llm_client`: Mock LLM client with configurable responses
- `mock_llm_responses`: Predefined LLM responses for different scenarios

#### Database Fixtures
- `mock_shared_memory`: Mock shared memory system
- `mock_vector_db`: Mock vector database with search capabilities

#### Agent Fixtures
- `mock_base_agent`: Mock base agent with common functionality

### Utility Fixtures

#### Data Generation
- `sample_code_files`: Sample code in different languages
- `benchmark_data`: Performance testing datasets
- `security_test_cases`: Security vulnerability test cases

#### Test Helpers
- `TestTimer`: Context manager for timing operations
- `MemoryTracker`: Memory usage tracking
- `MockLLMProvider`: Configurable mock LLM provider

## 🛠 Test Utilities

### Custom Assertions (`tests/utils/assertions.py`)

Domain-specific assertions for DevGuard components:

```python
# Memory and state assertions
assert_memory_entry_valid(entry)
assert_task_status_valid(task)
assert_agent_state_valid(state)

# LLM and response assertions
assert_llm_response_valid(response)
assert_mcp_response_valid(mcp_response)

# Security and quality assertions
assert_security_scan_result(scan_result)
assert_code_quality_metrics(metrics)
```

### Helper Functions (`tests/utils/helpers.py`)

Utility functions for test data generation and common operations:

```python
# Data generation
generate_random_string(length=10)
generate_test_code(language="python", complexity="simple")
create_mock_git_history(repo_path, num_commits=5)

# Test execution helpers
with TestTimer() as timer:
    # Timed operation
    pass

with MemoryTracker() as tracker:
    # Memory-tracked operation
    pass

# Environment management
with temporary_environment_variables(TEST_VAR="value"):
    # Test with specific environment
    pass
```

## 📊 Coverage and Quality

### Coverage Requirements

- **Unit Tests**: 95% minimum coverage
- **Integration Tests**: 85% minimum coverage
- **Overall**: 95% minimum coverage

### Quality Gates

All tests must pass these quality gates:

1. **Code Formatting**: Black and isort compliance
2. **Linting**: Ruff with strict rules
3. **Type Checking**: MyPy with strict mode
4. **Security**: Bandit and safety checks
5. **Performance**: No regressions in benchmarks

### Coverage Reports

Coverage reports are generated in multiple formats:

- **Terminal**: Summary with missing lines
- **HTML**: Detailed interactive report (`htmlcov/index.html`)
- **XML**: For CI/CD integration (`coverage.xml`)

## 🔄 Continuous Integration

### GitHub Actions Workflow

The CI pipeline runs:

1. **Quality Checks**: Formatting, linting, type checking
2. **Unit Tests**: Fast, isolated tests
3. **Integration Tests**: Component interaction tests
4. **Security Tests**: Vulnerability scanning
5. **Performance Tests**: Benchmark validation
6. **Build Tests**: Package building and installation

### Pre-commit Hooks

Automatic quality checks on every commit:

- Code formatting (black, isort)
- Linting (ruff)
- Type checking (mypy)
- Security scanning (bandit, safety)
- Test execution (pytest)

## 🎯 Test Markers

Use pytest markers to categorize and filter tests:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.performance   # Performance test
@pytest.mark.security      # Security test
@pytest.mark.slow          # Slow-running test
@pytest.mark.benchmark     # Benchmark test
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run all except slow tests
pytest -m "not slow"

# Run security and performance tests
pytest -m "security or performance"
```

## 🐛 Debugging Tests

### Verbose Output

```bash
# Detailed test output
pytest -v

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

### Test Debugging

```python
# Use pytest's built-in debugging
import pytest
pytest.set_trace()  # Breakpoint

# Or use standard pdb
import pdb
pdb.set_trace()
```

### Log Analysis

Test logs are available in:
- `logs/test.log`: General test logs
- `htmlcov/`: Coverage reports
- `junit-results.xml`: JUnit format results

## 📈 Performance Testing

### Benchmark Tests

Use pytest-benchmark for performance testing:

```python
def test_performance(benchmark):
    result = benchmark(function_to_test, arg1, arg2)
    assert result is not None
```

### Memory Profiling

```python
def test_memory_usage():
    with MemoryTracker() as tracker:
        # Memory-intensive operation
        pass
    
    assert tracker.peak_memory < 100  # MB
```

### Performance Thresholds

Set performance expectations:

```python
@pytest.mark.benchmark(
    min_rounds=5,
    max_time=1.0,
    warmup=True
)
def test_fast_operation(benchmark):
    benchmark(fast_function)
```

## 🔒 Security Testing

### Vulnerability Testing

Test for common security issues:

```python
def test_input_validation():
    with pytest.raises(ValidationError):
        process_user_input(malicious_input)

def test_sql_injection_prevention():
    result = safe_database_query(user_input)
    assert "DROP TABLE" not in result.query
```

### Security Scanning

Automated security scans:

- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner
- **Semgrep**: Static analysis security scanner

## 📝 Writing Good Tests

### Test Structure

Follow the AAA pattern:

```python
def test_example():
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_under_test(setup_data)
    
    # Assert
    assert result.success
    assert result.data == expected_data
```

### Test Naming

Use descriptive test names:

```python
def test_memory_entry_creation_with_valid_data():
    """Test that memory entries are created correctly with valid input data"""
    pass

def test_memory_entry_creation_fails_with_invalid_agent_id():
    """Test that memory entry creation fails when agent_id is invalid"""
    pass
```

### Test Documentation

Document complex test scenarios:

```python
def test_complex_agent_coordination():
    """
    Test complex agent coordination scenario:
    
    1. Git Watcher detects changes in repository A
    2. Impact Mapper identifies dependencies in repository B
    3. Planner creates tasks for both repositories
    4. Code Agent processes tasks in correct order
    5. QA Agent validates all changes
    
    This test verifies the complete workflow works correctly.
    """
    pass
```

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use editable install
pip install -e .
```

#### Fixture Not Found
```bash
# Check fixture is defined in conftest.py
# Ensure conftest.py is in correct location
# Verify fixture scope is appropriate
```

#### Slow Tests
```bash
# Run only fast tests
pytest -m "not slow"

# Use parallel execution
pytest -n auto
```

#### Memory Issues
```bash
# Monitor memory usage
pytest --memray

# Use memory profiler
python -m memory_profiler test_file.py
```

### Getting Help

1. Check test logs in `logs/`
2. Review coverage report in `htmlcov/`
3. Run tests with `-v` for verbose output
4. Use `pytest --collect-only` to see available tests
5. Check CI logs for detailed error information

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

---

For questions or issues with the testing infrastructure, please check the existing tests for examples or create an issue in the project repository.