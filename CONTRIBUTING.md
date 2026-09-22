# Contributing to DeSSIN

## License

By contributing, you agree your contributions are under the same license as this repository (**GNU AGPL v3 or later**). See `LICENSE`.

## Development setup

```bash
cd dessin
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests (default suite under `tests/`):

```bash
pytest tests/ -q
```

Slow or end-to-end jobs live under `tests/` / `e2e/` with markers; see `pyproject.toml` `[tool.pytest.ini_options]` and `docs/E2E_NETWORK_TESTS.md`.

## Style

Formatting and linting are configured via `pyproject.toml` (`black`, `ruff`, `isort`, `mypy`). Match surrounding code when touching files.
