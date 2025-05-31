.PHONY: format format-unsafe lint lint-unsafe setup test test-models test-core build publish

format:
	ruff format .
	ruff check . --fix

format-unsafe:
	ruff format .
	ruff check . --fix --unsafe-fixes

lint:
	ruff check $(if $(target),$(target),.)

lint-unsafe:
	ruff check . --unsafe-fixes

setup:
	uv venv .venv && . .venv/bin/activate && uv pip install --upgrade pip && uv pip install .[develop] build twine && exec $$SHELL

test:
	. .venv/bin/activate && pytest

test-models:
	. .venv/bin/activate && pytest tests/test_models/

test-core:
	. .venv/bin/activate && pytest tests/test_core/

test-gpt:
	. .venv/bin/activate && pytest tests/test_models/test_gpt.py

test-gemma:
	. .venv/bin/activate && pytest tests/test_models/test_gemma.py

test-claude:
	. .venv/bin/activate && pytest tests/test_models/test_claude.py

test-ask:
	. .venv/bin/activate && pytest tests/test_core/test_ask.py

test-validation:
	. .venv/bin/activate && pytest tests/test_core/test_validation.py

build:
	. .venv/bin/activate && python -m build

publish:
	. .venv/bin/activate && python -m twine upload dist/*
