PYTHON ?= python3

.PHONY: install install-lite run run-prod test lint check-kb

install:
	$(PYTHON) -m pip install -e ".[dev,hf]"

install-lite:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) scripts/run_server.py --mode dev

run-prod:
	$(PYTHON) scripts/run_server.py --mode prod

test:
	pytest

lint:
	ruff check .

check-kb:
	$(PYTHON) scripts/rebuild_index.py
