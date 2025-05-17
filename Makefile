.PHONY: format format-unsafe lint lint-unsafe setup test build publish

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

build:
	. .venv/bin/activate && python -m build

publish:
	. .venv/bin/activate && python -m twine upload dist/*
