VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup test download index eval

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q -e ".[dev]"

test:
	$(PY) -m ruff check .
	$(PY) -m pytest -q

download:
	$(PY) scripts/download_papers.py

index:
	$(PY) -m atlas.index

eval:
	@echo "eval: not implemented yet"

.PHONY: parse
parse:
	$(PY) scripts/parse_papers.py
