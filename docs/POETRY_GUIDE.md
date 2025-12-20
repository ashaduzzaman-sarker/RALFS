# Poetry Virtual Environment Guide

This guide explains how to use Poetry with RALFS for dependency management and virtual environment isolation.

## Overview

RALFS uses [Poetry](https://python-poetry.org/) for dependency management and packaging. Poetry provides:
- **Isolated virtual environments** - Dependencies are installed in a project-specific environment
- **Reproducible builds** - `poetry.lock` ensures consistent dependency versions
- **Easy dependency management** - Simple commands for adding/removing dependencies
- **Script execution** - Run commands within the virtual environment using `poetry run`

## Installation

### Install Poetry

```bash
# On Linux/macOS/WSL
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

For other installation methods, see [Poetry Installation Guide](https://python-poetry.org/docs/#installation).

## Basic Usage

### 1. Install Dependencies

```bash
# Install runtime dependencies (default)
poetry install

# Install with dev dependencies
poetry install --with dev

# Install without dev dependencies (explicit)
poetry install --without dev
```

### 2. Running Commands

All RALFS commands should be run with `poetry run` to ensure they execute within the virtual environment:

```bash
# Run RALFS CLI
poetry run ralfs --help
poetry run ralfs preprocess --dataset arxiv
poetry run ralfs build-index --dataset arxiv
poetry run ralfs train --dataset arxiv

# Run tests
poetry run pytest tests/

# Run linters
poetry run ruff check ralfs/
poetry run mypy ralfs/

# Format code
poetry run black ralfs/ tests/
poetry run isort ralfs/ tests/
```

### 3. Using the Makefile

The Makefile includes convenient targets that use Poetry:

```bash
# Install dependencies
make install          # Runtime dependencies
make install-dev      # Runtime + dev dependencies

# Run tests
make test             # Fast tests
make test-all         # All tests
make test-cov         # Tests with coverage

# Linting and formatting
make lint             # Run linters
make format           # Format code

# RALFS commands
make preprocess       # Preprocess data
make build-index      # Build indexes
make train            # Train model
make evaluate         # Run evaluation
```

### 4. Using Scripts

All scripts in the `scripts/` directory use Poetry:

```bash
# These scripts automatically use poetry run
bash scripts/preprocess.sh arxiv train 100
bash scripts/build_index.sh arxiv
bash scripts/train.sh arxiv configs/train/default.yaml
bash scripts/evaluate.sh results/predictions.json data/test/references.json
bash scripts/pipeline.sh arxiv 100
```

## Managing Dependencies

### Add a New Dependency

```bash
# Add runtime dependency
poetry add package-name

# Add dev dependency
poetry add --group dev package-name

# Add with version constraint
poetry add "package-name>=1.0.0,<2.0.0"
```

### Remove a Dependency

```bash
poetry remove package-name
```

### Update Dependencies

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update package-name

# Show outdated packages
poetry show --outdated
```

## Working with the Virtual Environment

### Activate the Shell

```bash
# Spawn a shell within the virtual environment
poetry shell

# Now you can run commands directly without 'poetry run'
ralfs --help
pytest tests/
```

### Check Virtual Environment Location

```bash
poetry env info
```

### Remove Virtual Environment

```bash
poetry env remove python
```

## Troubleshooting

### Poetry Not Found

If `poetry` command is not found:

1. Ensure Poetry is installed: `curl -sSL https://install.python-poetry.org | python3 -`
2. Add to PATH: `export PATH="$HOME/.local/bin:$PATH"`
3. Restart your shell or run: `source ~/.bashrc` (or `~/.zshrc`)

### Python Version Mismatch

RALFS requires Python 3.10-3.12. Check your Python version:

```bash
python3 --version
```

If needed, specify the Python version for Poetry (use 3.10, 3.11, or 3.12):

```bash
# Example: Use Python 3.10
poetry env use python3.10

# Or use Python 3.11
poetry env use python3.11

# Or use Python 3.12
poetry env use python3.12

# Then install dependencies
poetry install
```

### Lock File Out of Sync

If you see "lock file is out of sync" error:

```bash
poetry lock --no-update
poetry install
```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Try updating the lock file
poetry lock

# Or force reinstall
poetry env remove python
poetry install
```

## Best Practices

1. **Always use `poetry run`** or activate the shell with `poetry shell` before running commands
2. **Commit `poetry.lock`** to version control for reproducible builds
3. **Don't modify `pyproject.toml` manually** for dependencies - use `poetry add/remove`
4. **Use virtual environments** - Poetry manages this automatically
5. **Keep Poetry updated** - Run `poetry self update` periodically

## CI/CD Integration

In CI/CD pipelines (GitHub Actions, GitLab CI, etc.):

```yaml
# Example GitHub Actions workflow
- name: Install Poetry
  run: |
    curl -sSL https://install.python-poetry.org | python3 -
    echo "$HOME/.local/bin" >> $GITHUB_PATH

# Or use the official Poetry action
- name: Install Poetry (Alternative)
  uses: snok/install-poetry@v1
  with:
    version: latest
    virtualenvs-create: true
    virtualenvs-in-project: true

- name: Install dependencies
  run: poetry install

- name: Run tests
  run: poetry run pytest tests/

- name: Run linters
  run: |
    poetry run ruff check ralfs/
    poetry run mypy ralfs/
```

## Additional Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry Commands Reference](https://python-poetry.org/docs/cli/)
- [Managing Dependencies](https://python-poetry.org/docs/managing-dependencies/)
- [Virtual Environments](https://python-poetry.org/docs/managing-environments/)
