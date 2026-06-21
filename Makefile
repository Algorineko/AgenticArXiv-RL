.PHONY: start stop restart logs install

start:
	@./bin/start.sh --logs

stop:
	@./bin/shutdown.sh

restart: stop start

logs:
	@tail -f .run/backend.log .run/frontend.log

install:
	python3 -m venv .venv
	.venv/bin/pip install -r AgenticArxiv/requirements.txt
	cd AgenticArxivWeb && npm install
