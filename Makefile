.PHONY: install run test lint

install:
	python -m pip install -e ".[dev]"

run:
	.venv/bin/python -m streamlit run app.py

test:
	pytest -q

lint:
	ruff check .
