# One command per stage (docs/PLAN.md, "Repository and tooling").
# Data: `make data` fetches everything; `make data ONLY="uatd noaa"` fetches a subset.
PY := uv run python
ONLY ?=

.PHONY: setup data data-dry test lint check

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
