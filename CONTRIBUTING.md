# Contributing to Advanced Data Toolkit

First off, thank you for considering contributing to Advanced Data Toolkit! It's people like you that make this tool better for everyone.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct: be respectful, inclusive, and constructive.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (sample data files, screenshots)
- **Describe the behavior you observed and what you expected**
- **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed functionality**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Make your changes** following our coding style
4. **Add tests** for any new functionality
5. **Ensure all tests pass**: `pytest`
6. **Format your code**: `black src/`
7. **Update documentation** if needed
8. **Submit your pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/advanced-data-toolkit.git
cd advanced-data-toolkit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

## Project Structure

```
advanced-data-toolkit/
├── src/
│   └── data_toolkit/
│       ├── __init__.py
│       ├── main_gui.py           # Main GUI application
│       ├── data_loading_methods.py
│       ├── statistical_analysis.py
│       ├── ml_models.py
│       ├── bayesian_analysis.py
│       ├── uncertainty_analysis.py
│       ├── nonlinear_analysis.py
│       ├── timeseries_analysis.py
│       ├── causality_analysis.py
│       └── visualization_methods.py
├── tests/                        # Test files
├── notebooks/                    # Jupyter notebooks
├── examples/                     # Example scripts
├── docs/                         # Documentation
├── pyproject.toml
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

## Coding Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible
- Use type hints for function signatures

### Example Function Style

```python
def calculate_correlation(
    x: np.ndarray, 
    y: np.ndarray, 
    method: str = 'pearson'
) -> Tuple[float, float]:
    """
    Calculate correlation between two arrays.
    
    Args:
        x: First array of values
        y: Second array of values
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple of (correlation coefficient, p-value)
        
    Raises:
        ValueError: If arrays have different lengths
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have the same length")
    
    # Implementation...
```

## Testing

- Write tests for all new functionality
- Place tests in the `tests/` directory
- Name test files as `test_<module>.py`
- Use pytest fixtures for common setup

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=data_toolkit

# Run specific test file
pytest tests/test_statistical_analysis.py
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Include usage examples in docstrings
- Update notebooks if adding new features

## Questions?

Feel free to open an issue with the "question" label if you have any questions about contributing.

Thank you for your contributions! 🎉
