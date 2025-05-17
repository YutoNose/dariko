.PHONY: format format-unsafe lint lint-unsafe

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
