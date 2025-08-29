# Contributing to bijx

If you're interested in contributing or using this library, feel free to reach out!

## Development Setup

1. Clone and install, including dependencies for testing and building documentation:
   ```bash
   git clone https://github.com/mathisgerdes/bijx.git
   cd bijx
   pip install -e ".[dev,docs]"
   ```

2. To keep the code tidy and consistent, install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. After making edits, pre-commit tools can be run manually (optional; otherwise they will run automatically on commit):
   ```bash
   # Run all hooks on all files
   pre-commit run --all-files

   # Run specific hooks on all files
   pre-commit run black --all-files
   pre-commit run ruff --all-files

   # Run all hooks on specific files or directories
   pre-commit run --files src/bijx/some_file.py
   pre-commit run --files src/bijx/some_directory/

   # Run all hooks on staged files only
   pre-commit run
   ```

## Testing

- Run tests: `pytest tests/`
- Run doctests: `pytest src/bijx/ --doctest-modules`
- Run all: `pytest tests/ src/bijx/ --doctest-modules`
- Parallel execution: `pytest -n auto`

This project uses property-based testing with Hypothesis to ensure mathematical correctness of bijections.

## Pull Requests

General process:

1. Fork the repository on GitHub
2. Create a feature branch for your changes
3. Make your changes following best practices:
   - Write tests for new functionality
   - Ensure all tests pass and code quality checks succeed (see above)
   - Update documentation if needed
4. Commit & Push to your fork
5. Submit a pull request

## Issues

- **Bug reports**: Include minimal reproducible examples and describe your setup
- **Feature requests**: Describe the use case and expected behavior
- **Questions**: Use discussions instead of issues

## Documentation

Make sure code blocks in doc-strings are properly formatted and can be executed as tests.

- API documentation is built with Sphinx
- Build docs: `cd docs && make html`
- Serve docs locally: `cd docs && make livehtml`

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
