.PHONY: install install-extra run

install:
	python3 -m venv .venv
	.venv/bin/pip install -r AgenticArxiv/requirements.txt

install-extra: install
	.venv/bin/pip install -r AgenticArxiv/requirements-extra.txt

run:
	.venv/bin/python AgenticArxiv/main.py
