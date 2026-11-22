.PHONY: install install-serve check-venv test test-all validate-cfr validate-coverage validate-lp

# Every Python entry point runs through $(PY): the project interpreter with this checkout's src/ and
# root first on the path, so the code under test is always this checkout's.
PY ?= PYTHONPATH=$(CURDIR)/src:$(CURDIR) .venv/bin/python

PYTEST_WORKERS ?= 6

# Single-threaded numeric libraries for the test suite.
TEST_ENV = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1

install:
	uv sync --extra dev --extra tracking

# vLLM is a heavy, GPU-specific dependency kept out of the default install.
install-serve:
	uv sync --extra dev --extra tracking --extra serve

check-venv:
	@test -x .venv/bin/python || { echo "ERROR: no interpreter at $(CURDIR)/.venv/bin/python; run: make install"; exit 1; }

# The core test set, without the slow markers.
test: check-venv
	$(TEST_ENV) $(PY) -m pytest -n $(PYTEST_WORKERS) -m "not slow and not coverage_run"

# The core test set including the slow markers.
test-all: check-venv
	$(TEST_ENV) $(PY) -m pytest -n $(PYTEST_WORKERS)

# The scientific-validity gate: solves CFR+ per game and exits non-zero on a tolerance breach or if
# Kuhn departs from -1/18.
validate-cfr: check-venv
	$(PY) scripts/validate_cfr.py

# The semi-synthetic coverage and mutation checks (slow).
validate-coverage: check-venv
	$(TEST_ENV) $(PY) -m pytest -n $(PYTEST_WORKERS) -m coverage_run

# Cross-check of the references against a linear-program solution.
validate-lp: check-venv
	$(PY) scripts/validate_lp.py $(LP_ARGS)
