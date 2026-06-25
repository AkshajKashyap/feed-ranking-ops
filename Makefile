.PHONY: install test lint check

install:
python -m pip install -e ".[dev]"

test:
pytest -q

lint:
ruff check .

check: lint test
