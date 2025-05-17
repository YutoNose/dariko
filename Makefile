.PHONY: format format-unsafe lint lint-unsafe setup test

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
	uv venv .venv && . .venv/bin/activate && uv pip install .[develop] && exec $$SHELL

test:
	. .venv/bin/activate && pytest
