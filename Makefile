.PHONY: test lint check

test:
	pytest

lint:
	ruff check src tests scripts

check: test lint
