.PHONY: install test lint prepare train evaluate predict docker-build docker-run clean

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

lint:
	ruff check src tests

prepare:
	dna-sentinel prepare --config config/cassiopeia_prime.yaml
	dna-sentinel prepare-features --config config/cassiopeia_prime.yaml

train:
	dna-sentinel train --config config/cassiopeia_prime.yaml

evaluate:
	dna-sentinel benchmark \
	  --checkpoint artifacts/cassiopeia_prime/cassiopeia_best.pt \
	  --data-dir data/dna_sentinel

predict:
	dna-sentinel predict --checkpoint artifacts/cassiopeia_prime/cassiopeia_best.pt --fasta data/dna_sentinel/query.fa --json

docker-build:
	docker build -t dna-sentinel .

docker-run:
	docker run --rm -p 8000:8000 dna-sentinel

clean:
	rm -rf .pytest_cache .ruff_cache
