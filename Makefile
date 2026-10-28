.PHONY: install train test check
install:
	python -m pip install -e ".[dev]"
train:
	python -m digits_cnn train
test:
	python -m pytest -q
check:
	python -m ruff check .
	python -m ruff format --check .
