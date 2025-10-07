# Development Guide

## Setting Up Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/dot-gabriel-ferrer/universal-dng-converter.git
cd universal-dng-converter
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

## Code Quality Tools

### Formatting with Black

```bash
black src tests
```

### Linting with Flake8

```bash
flake8 src tests
```

### Type Checking with MyPy

```bash
mypy src
```

### Import Sorting with isort

```bash
isort src tests
```

## Testing

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_converter.py
```

### Run Tests in Multiple Python Versions

```bash
tox
```

## Project Structure

```
universal-dng-converter/
├── src/
│   └── universal_dng_converter/
│       ├── __init__.py
│       ├── converter.py      # Main converter class
│       └── cli.py           # Command-line interface
├── tests/
│   ├── __init__.py
│   └── test_converter.py    # Test suite
├── scripts/
│   └── convert-to-dng       # Standalone script
├── docs/
│   ├── README.md
│   ├── installation.md
│   ├── usage.md
│   └── development.md
├── examples/
│   └── basic_usage.py       # Usage examples
├── pyproject.toml           # Project configuration
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Development dependencies
├── .gitignore
├── .pre-commit-config.yaml
├── tox.ini
└── README.md
```

## Contributing Guidelines

### 1. Fork and Branch

1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature/my-feature`

### 2. Make Changes

1. Write your code following the existing style
2. Add tests for new functionality
3. Update documentation as needed

### 3. Quality Checks

1. Run the test suite: `pytest`
2. Check code formatting: `black --check src tests`
3. Run linting: `flake8 src tests`
4. Check types: `mypy src`

### 4. Submit Pull Request

1. Commit your changes: `git commit -m "Add my feature"`
2. Push to your fork: `git push origin feature/my-feature`
3. Open a Pull Request on GitHub

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 style guidelines
- Use Black for automatic formatting
- Maximum line length: 88 characters
- Use type hints for all public APIs

### Documentation Style

- Use clear, concise language
- Include code examples for public APIs
- Document all parameters and return values
- Use Google-style docstrings

### Testing Guidelines

- Write tests for all new functionality
- Aim for >90% code coverage
- Use descriptive test names
- Group related tests in classes

## Release Process

### 1. Version Bumping

Update version in:
- `src/universal_dng_converter/__init__.py`
- `pyproject.toml`

### 2. Update Changelog

Document changes in `CHANGELOG.md`

### 3. Create Release

```bash
# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

### 4. Tag Release

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Architecture Overview

### Core Components

1. **DNGImageConverter**: Main converter class handling format detection and conversion
2. **CLI Module**: Command-line interface implementation
3. **Format Handlers**: Specialized handlers for different input formats

### Key Design Principles

- **Modularity**: Each format handler is independent
- **Extensibility**: Easy to add new input formats
- **Error Handling**: Robust error handling and logging
- **Performance**: Efficient memory usage for large images
- **Compatibility**: Works across different Python versions and platforms
