.PHONY: install install-local install-train install-translate install-dev run check

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e .

install-local: install
	.venv/bin/python -m pip install -e '.[local]'

install-train: install
	.venv/bin/python -m pip install -e '.[train]'

install-translate: install
	.venv/bin/python -m pip install -e '.[translate]'

install-dev: install
	.venv/bin/python -m pip install -e '.[dev]'

run:
	.venv/bin/agentic-arxiv run

check:
	.venv/bin/python -m pytest tests/test_architecture.py tests/test_install_metadata.py tests/test_config_resources.py
