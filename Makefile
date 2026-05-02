PYTHON ?= python3

.PHONY: test test-cov lint typecheck security verify release-verify run compile quality install-dev smoke-cli clean-reports

test:
	$(PYTHON) -m pytest -q tests

test-cov:
	$(PYTHON) -m pytest -q tests --cov=src/bio_spread_project --cov-report=term-missing --cov-fail-under=90

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

install-dev:
	$(PYTHON) -m pip install -c constraints.txt -e .[dev]

smoke-cli:
	PYTHONPATH=src $(PYTHON) -m bio_spread_project.cli run --mode input --input data/sample_plasmid_records.csv --output-dir reports/smoke_cli

clean-reports:
	rm -rf final_report_demo final_report_production audit_demo
	rm -rf reports/cli_run reports/run reports/smoke_cli output/playwright
	rm -f reports/.DS_Store

security:
	$(PYTHON) -m pip_audit -r requirements.txt

quality: lint typecheck test-cov security

verify:
	$(PYTHON) -m bio_spread_project.cli verify

release-verify:
	$(PYTHON) -m bio_spread_project.cli verify --release

run:
	$(PYTHON) -m bio_spread_project.cli run

compile:
	$(PYTHON) -m compileall src
