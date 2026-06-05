.PHONY: install test lint prepare train evaluate predict serve web docker-build docker-run clean

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest -q

test-cov:
	python3 -m pytest --cov=src --cov-report=term

lint:
	ruff check src tests

typecheck:
	python3 -m mypy src --ignore-missing-imports

prepare:
	dna prep -c config/cassiopeia_prime.yaml

features:
	dna features -c config/cassiopeia_prime.yaml

train:
	dna train -c config/cassiopeia_prime.yaml

evaluate:
	dna bench -m artifacts/cassiopeia_prime_v14/cassiopeia_best.pt -d data/dna_sentinel -o artifacts/cassiopeia_prime_v14/report.json

predict:
	dna predict -m artifacts/cassiopeia_prime_v14/cassiopeia_best.pt -f data/dna_sentinel/query.fa --interpret

interpret:
	dna interpret -m artifacts/cassiopeia_prime_v14/cassiopeia_best.pt -f data/dna_sentinel/query.fa

serve:
	dna serve -m artifacts/cassiopeia_prime_v14/cassiopeia_best.pt

web:
	python3 -m http.server 4173 --directory web

docker-build:
	docker build -t dna-sentinel .

docker-run:
	docker run --rm -p 8000:8000 dna-sentinel

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov
