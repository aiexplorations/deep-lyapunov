# Installation

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher

## Basic Installation

Install from PyPI:

```bash
pip install deep-lyapunov
```

## Installation with Extras

### Full Installation

Includes additional visualization features:

```bash
pip install deep-lyapunov[full]
```

### Development Installation

For contributing or running tests:

```bash
pip install deep-lyapunov[dev]
```

### Documentation

To build documentation locally:

```bash
pip install deep-lyapunov[docs]
```

### Everything

Install all optional dependencies:

```bash
pip install deep-lyapunov[all]
```

## From Source

For the latest development version:

```bash
git clone https://github.com/aiexplorations/deep-lyapunov.git
cd deep-lyapunov
pip install -e ".[dev]"
```

## Verifying Installation

```python
import deep_lyapunov
print(deep_lyapunov.__version__)
```

Expected output: `0.1.0`

## Troubleshooting

### CUDA Support

If you want to use GPU acceleration, ensure you have a CUDA-compatible PyTorch installation:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Common Issues

**ImportError: No module named 'deep_lyapunov'**

Ensure the package is installed in your active Python environment:

```bash
pip list | grep deep-lyapunov
```

**PyTorch version mismatch**

deep-lyapunov requires PyTorch 2.0+. Upgrade if needed:

```bash
pip install --upgrade torch
```
