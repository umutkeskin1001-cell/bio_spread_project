.PHONY: install train test lint evaluate prepare clean benchmark tune docker-build docker-run

install:
	pip install -e .

train:
	python3 -m bio_spread.cli.main train --config config/default.yaml

test:
	python3 -m pytest tests/ -v

lint:
	ruff check src/ && ruff format --check src/

evaluate:
	python3 -m bio_spread.cli.main evaluate --model-path $(MODEL) --config config/default.yaml --feature-dir data/features

prepare:
	python3 -m bio_spread.cli.main prepare --config config/default.yaml

clean:
	rm -rf artifacts/ .pytest_cache .ruff_cache __pycache__

benchmark:
	bio-spread train --config config/benchmark.yaml

tune:
	python scripts/hyperparameter_tuning.py

docker-build:
	docker build -t bio-spread .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data/features:/app/data/features bio-spread
