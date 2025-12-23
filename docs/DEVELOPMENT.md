# Development Environment Setup

## Overview
This project uses C++ scheduling algorithms with Python bindings via pybind11, integrated into a FastAPI backend.

## Environment Setup

### 1. Python Environment
- **Recommended**: Use conda or virtual environment
- **Python version**: >=3.11
- **Important**: Install in the same environment where you'll run the application

```bash
# Using conda (recommended)
conda create -n scheduler-dev python=3.11
conda activate scheduler-dev

# OR using venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### 2. Install in Editable Mode
```bash
# Install project in development/editable mode
pip install -e .
```

**What this does:**
- Builds C++ extensions (scheduler_cpp module)
- Installs Python dependencies
- Creates editable link to your source code
- Python code changes are immediately available
- C++ changes require reinstall

### 3. Verify Installation
```bash
# Test C++ module import
python -c "import scheduler_cpp; print('✅ C++ module works')"

# Test API server
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
# Visit: http://127.0.0.1:8001/docs
```

## Development Workflow

### Python Code Changes
- Edit Python files directly
- Changes are immediately available (no reinstall needed)
- Server auto-reloads with `--reload` flag

### C++ Code Changes
1. Modify C++ files in `cpp/` directory
2. Rebuild and reinstall:
```bash
pip install -e .
```
3. Restart server if running

### Dependencies
- **Add new Python deps**: Edit `pyproject.toml` → `pip install -e .`
- **Local development**: Uses editable install, not PyPI package
- **Self-dependency**: The `os-scheduler-atlas==0.1.0` dependency is satisfied by local editable install

## Common Issues

### ModuleNotFoundError: scheduler_cpp
**Cause**: C++ module not built or installed
**Fix**:
```bash
pip install -e .
```

### Wrong Python Environment
**Cause**: Module installed in different environment than where you're running
**Check**:
```bash
pip list | grep os-scheduler-atlas
# Should show: os-scheduler-atlas 0.1.0 /path/to/your/project
```

### Build Failures
**Cause**: Missing build dependencies
**Fix**:
```bash
# Ensure build tools available
pip install scikit-build-core pybind11 setuptools wheel
```

## Project Structure
```
backend/
├── src/
│   ├── __init__.py       # Required for Python package
│   ├── api/
│   │   ├── __init__.py   # Required for API package
│   │   └── main.py       # FastAPI application
│   └── ml/
│       ├── __init__.py   # Required for ML package
│       └── *.py          # ML modules
├── cpp/
│   ├── CMakeLists.txt    # Main CMake config
│   └── scheduler/
│       ├── CMakeLists.txt # Scheduler CMake config
│       ├── scheduler.cpp  # C++ implementation
│       └── pybind_module.cpp # Python bindings
└── pyproject.toml        # Project configuration
```

## PyPI vs Local Development
- **Local development**: Use `pip install -e .` (editable mode)
- **PyPI package**: For external users, not local development
- **Version conflicts**: Avoided by editable install taking precedence
- **Self-dependency**: Satisfied by local editable install, not PyPI download