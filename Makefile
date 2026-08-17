.PHONY: help test test-all test-fast lint types cov gate gate-update verify clean-cov

PY := .venv/bin/python
COV := --cov=src/nifty_quant --cov-branch

help:
	@echo "test         - fast suite (slow tests deselected)"
	@echo "test-all     - full suite including slow/real-data tests"
	@echo "lint         - ruff check on src and tests"
	@echo "types        - mypy on src"
	@echo "cov          - fast suite with a per-module coverage report"
	@echo "gate         - THE GATE: lint + types + coverage ratchet. Run this before claiming done."
	@echo "gate-update  - raise coverage_floor.json to currently-measured coverage (never lowers)"
	@echo "verify       - gate + the verification tier + the volume_breakout regression backtest"

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

test-fast: test

lint:
	$(PY) -m ruff check src tests

# mypy is on a RATCHET, not a pass/fail gate: HEAD carries 162 pre-existing errors across 21
# files (mostly numpy typing in strategy/plugins/, despite an earlier progress note claiming
# "mypy clean on src"). Blocking on zero would block all work; ignoring it entirely lets the
# count grow. So: the count may go DOWN but never UP. Lower MYPY_MAX as debt is paid.
MYPY_MAX := 162

types:
	@n=$$($(PY) -m mypy src 2>&1 | grep -cE '^src/.*: error:' || true); \
	echo "mypy errors: $$n (ceiling $(MYPY_MAX))"; \
	if [ "$$n" -gt "$(MYPY_MAX)" ]; then \
	  echo "FAIL: mypy error count rose from $(MYPY_MAX) to $$n. Fix the new errors or justify."; \
	  $(PY) -m mypy src | grep -E '^src/.*: error:' | head -40; \
	  exit 1; \
	fi; \
	if [ "$$n" -lt "$(MYPY_MAX)" ]; then \
	  echo "Ratchet: lower MYPY_MAX in the Makefile to $$n."; \
	fi

types-full:
	$(PY) -m mypy src

cov:
	$(PY) -m pytest -q -m "not slow" $(COV) --cov-report=term-missing

# The ratchet. Coverage thresholds are per-module and live in coverage_floor.json; they are
# enforced here rather than via a global fail_under in pyproject.toml, so that ad-hoc `--cov`
# runs during development stay exit-0 and the gate stays a deliberate, visible act.
gate: lint types
	$(PY) -m pytest -q -m "not slow" $(COV) --cov-report=json:.coverage.json --cov-report=term
	$(PY) scripts/coverage_gate.py --report .coverage.json --floor coverage_floor.json

gate-update:
	$(PY) -m pytest -q -m "not slow" $(COV) --cov-report=json:.coverage.json
	$(PY) scripts/coverage_gate.py --report .coverage.json --floor coverage_floor.json --update

# The verification tier is the adversarial suite (causality probes, leak canaries, noise nulls,
# parameter-sensitivity nulls). The final backtest is a regression guard: volume_breakout on
# 2024 is the repo's reference result and must not drift silently.
verify: gate
	$(PY) -m pytest tests/verification -v
	$(PY) -m nifty_quant.cli backtest --strategy volume_breakout \
		--config configs/strategies/volume_breakout.yaml \
		--start 2024-01-01 --end 2024-12-31

clean-cov:
	rm -f .coverage .coverage.json
