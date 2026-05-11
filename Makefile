.PHONY: install train test predict lint clean

install:
	pip install -e .

train:
	python3 -m bio_spread_reborn.cli.main train --config config/default.yaml

test:
	pytest tests/

lint:
	ruff check src/

predict:
	python3 -m bio_spread_reborn.cli.main predict --input-path $(INPUT) --output-path $(OUTPUT)

evaluate:
	python3 -m bio_spread_reborn.cli.main evaluate --input-path $(INPUT)

clean:
	rm -rf *.pt predictions.json .pytest_cache .ruff_cache
