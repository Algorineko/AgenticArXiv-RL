# Contributing to AgenticArXiv-RL

Thank you for your interest in contributing to AgenticArXiv-RL! This document outlines the process for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/AgenticArXiv-RL.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Create a virtual environment, install `pip install -e '.[dev]'`, and make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "feat: your feature description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized
- Use the installed `agentic_arxiv` package; do not add `PYTHONPATH`, `sys.path` injection, or aliases for the retired `AgenticArxiv` namespace
- Keep runtime independent of task catalogs, scoring free of tool/file execution, and evaluation independent of training callbacks

## Installation and Commands

Dependencies and extras are declared only in [pyproject.toml](pyproject.toml). Base `pip install .` supports API workflows without local-model or training dependencies. Add `pip install '.[local]'` for local inference, `pip install '.[train]'` for SFT/GRPO and training-side regressions, `pip install '.[translate]'` for pdf2zh, or `pip install '.[logging]'` for tensorboard. The dev extra does not implicitly install train.

Use `agentic-arxiv` or the same CLI through `python -m agentic_arxiv`. Root and subcommand `--help` must remain usable without training dependencies, model loading, API calls, credentials, or log creation. Global `--workspace` precedes the subcommand and defaults to the current directory; relative input/output paths resolve against that workspace. Default training configuration is a package resource, not a path into the source checkout. See [the architecture migration guide](docs/architecture_migration.md) for command and source mappings.

## Testing

- All new features must include tests
- Run the applicable suite before submitting: `pytest tests/`; report missing optional dependencies and which training-side checks were not run
- Tests should be deterministic and not require external API keys
- Test discovery is rooted at `tests/`; archived behavior baselines are not active tests or acceptance evidence
- Verify both editable and normal wheel installs outside the source directory for packaging or CLI changes; a complete development environment does not prove base-install isolation
- Do not generate research data, call real APIs, load tokenizers/models, or launch training merely to validate architecture changes; keep those unverified integration scopes explicit
- Preserve historical manifests, snapshots, trajectories, reports, and archived OpenSpec text at their original paths and contents. Record source migrations separately rather than rewriting old fingerprints or commands

## Pull Request Process

1. Ensure your PR description clearly explains the changes and their motivation
2. Reference any related issues
3. Keep PRs focused on a single change
4. Be responsive to review feedback
5. All CI checks must pass

## Reporting Issues

When reporting issues, please include:
- A clear and descriptive title
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Your environment (Python version, OS, etc.)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

Feel free to open an issue or start a discussion if you have questions about contributing.
