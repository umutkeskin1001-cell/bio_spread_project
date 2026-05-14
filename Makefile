.PHONY: install train test lint evaluate prepare clean

install:
	pip install -e .

train:
	python3 -m bio_spread_reborn.cli.main train --config config/default.yaml

test:
	python3 -m pytest tests/ -v

lint:
	ruff check src/ && ruff format --check src/

evaluate:
	python3 -m bio_spread_reborn.cli.main evaluate --model-path $(MODEL) --config config/default.yaml --feature-dir data/sovereign_features

prepare:
	python3 -m bio_spread_reborn.cli.main sovereign-prepare --config config/default.yaml

clean:
	rm -rf artifacts/ .pytest_cache .ruff_cache __pycache__
