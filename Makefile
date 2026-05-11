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
	python3 -m bio_spread_reborn.cli.main predict --input-path $(INPUT) --output-path $(OUTPUT) --model-path $(MODEL) --tokenizer-path $(TOKENIZER)

evaluate:
	python3 -m bio_spread_reborn.cli.main evaluate --input-path $(INPUT) --model-path $(MODEL) --tokenizer-path $(TOKENIZER)

snapshot:
	python3 -m bio_spread_reborn.cli.main snapshot --config config/default.yaml

clean:
	rm -rf *.pt predictions.json .pytest_cache .ruff_cache artifacts/
