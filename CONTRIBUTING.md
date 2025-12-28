# Contributing to deep-lyapunov

Thank you for your interest in contributing to deep-lyapunov!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aiexplorations/deep-lyapunov.git
   cd deep-lyapunov
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=deep_lyapunov --cov-report=term-missing

# Run specific test file
pytest tests/test_analyzer.py -v
```

## Code Style

We use:
- **black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking

Format your code before committing:

```bash
black src tests
isort src tests
ruff check src tests --fix
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/`
6. Format your code: `black src tests && isort src tests`
7. Commit with a descriptive message
8. Push to your fork and create a Pull Request

## Commit Messages

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Docs: documentation changes`
- `Test: test additions/changes`
- `Refactor: code refactoring`

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Minimal code to reproduce the issue
- Full error traceback

## Questions?

Feel free to open an issue for questions or discussions.
