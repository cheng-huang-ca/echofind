# One command per stage (docs/PLAN.md, "Repository and tooling").
# Data: `make data` fetches everything; `make data ONLY="uatd noaa"` fetches a subset.
PY := uv run python
ONLY ?=

.PHONY: setup data data-dry test lint check atlas

setup:            ## install the core package plus dev tools
	uv sync --group dev

data:             ## fetch datasets into data/raw (idempotent, resumable)
	$(PY) -m echofind.data.fetch $(if $(ONLY),--only $(ONLY))

data-dry:         ## show what `make data` would fetch, with sizes
	$(PY) -m echofind.data.fetch --dry-run $(if $(ONLY),--only $(ONLY))

test:
	uv run pytest

lint:
	uv run ruff check src tests

check: lint test

atlas:            ## W1 signal atlas from NOAA and MBARI (needs `make data ONLY="noaa mbari"`, echopype)
	uv pip install "echopype>=0.9"
	$(PY) scripts/w1_noaa.py
	$(PY) scripts/w1_mbari.py
	$(PY) scripts/w1_figures.py
