# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**python-simple-logger** is a Python logging library providing colored console/file logging with duplicate log filtering and sensitive data masking. Built on top of `colorlog`, it extends Python's standard logging with custom log levels (SUCCESS, HASH, STEP) and filters.

## Development Commands

### Testing

```bash
# Run tests across all Python versions (3.9, 3.10, 3.11, 3.12, 3.13)
tox

# Run tests for current environment only
uv run pytest simple_logger/tests

# Run tests with coverage (auto-configured via pytest.ini)
uv run pytest simple_logger/tests -v
```

**Coverage Requirements:**
- Minimum coverage: 93% (enforced in pyproject.toml)
- Coverage report: HTML output in `.tests_coverage/`
- Test files excluded from coverage (see `tool.coverage.run.omit`)

### Code Quality

```bash
# Format code (ruff auto-fix enabled)
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy simple_logger/
```

**Quality Standards:**
- Line length: 120 characters
- Type hints required (mypy strict mode: disallow_untyped_defs, disallow_incomplete_defs)
- Ruff preview mode enabled with auto-fix

## Architecture

### Core Components

**`simple_logger/logger.py`** - Single-file library with four main components:

1. **`SimpleLogger`** (lines 60-79)
   - Custom logger class extending Python's `logging.Logger`
   - Adds custom log levels: SUCCESS (32), HASH (33), STEP (34)
   - Provides convenience methods: `.success()`, `.step()`, `.hash()`
   - Set as default logger class via `logging.setLoggerClass(SimpleLogger)` (line 82)

2. **`WrapperLogFormatter`** (lines 55-57)
   - Extends `colorlog.ColoredFormatter`
   - Overrides `formatTime()` to use ISO 8601 timestamps
   - Configured with color mapping for all log levels

3. **Filters**
   - `DuplicateFilter` (lines 18-37): Suppresses repeated logs, appends count summary
   - `RedactingFilter` (lines 40-52): Masks sensitive patterns with regex

4. **`get_logger()`** (lines 85-166)
   - Factory function returning singleton loggers (cached in `LOGGERS` dict)
   - Handles both console and file handlers with rotation
   - Auto-detects `FORCE_COLOR` env var for non-TTY environments (Docker/CI)

### Key Design Patterns

**Singleton Logger Pattern:**
```python
if LOGGERS.get(name):
    return LOGGERS[name]  # Return cached logger
```
Ensures one logger instance per name.

**Force Color Detection (lines 119-123):**
```python
if force_color is None:
    force_color_env = os.environ.get("FORCE_COLOR", "").lower()
    force_color = force_color_env in ("1", "true")
```
Critical for Docker logs - passes `force_color` to ColoredFormatter to enable ANSI codes in non-TTY environments.

## Important Implementation Details

### Adding New Features

When modifying `get_logger()`:
- Parameters must be added to function signature
- Update docstring with new parameter
- Consider singleton behavior - changes affect all future logger creations, not existing ones
- Update README.md with usage examples

### Color Output

The `force_color` parameter/env var is essential for Docker environments where `sys.stdout.isatty()` returns False. Without it, colorlog disables colors even though Docker can display ANSI codes.

### Testing

Tests use pytest fixtures for logger instances (`basic_logger`, `logger_with_mask`). When adding features:
- Test both parameter and environment variable paths
- Use `tmp_log_file` fixture for file output tests
- Verify both console and file handler behavior

## Release Process

This project uses `release-it` for versioning and releases:

```bash
# Prerequisites (one-time setup)
export GITHUB_TOKEN=<token>
sudo npm install --global release-it

# Create release
git pull
release-it  # Interactive prompts guide the process
```

Version is managed in `pyproject.toml` (currently 2.0.17).

## Build System

- **Package manager:** uv (with lock file)
- **Build backend:** hatchling
- **Distribution:** PyPI at `python-simple-logger`
- **Include:** Only `simple_logger/` directory in builds
