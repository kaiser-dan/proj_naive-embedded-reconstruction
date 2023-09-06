.DEFAULT_GOAL: all

# --- Help ---
.PHONY: help
help:
	$(MAKE) --print -data-base --question | \
	$(AWK) '/^[^.%][-A-Za-z0-9_]*:/	{ print substr($$1, 1, length($$1)-1) }' | \
	sort | \
	$(PR) --omit-pagination --width=80 --columns=4


# --- Default target ---
.PHONY: all
all: build check


# --- Package compilation & testing ---
.PHONY: build install check
build: install
	pip install .[test]
	pip install .[workflow]

install:
	pip install .

check:
	pytest -v

# --- Cleaning ---
.PHONY: clean deepclean
clean:
	@echo "Removing log files..."
	@find ./ -type f -name "*.log" -delete
	@echo "Removing temp directory..."
	@find ./ -type d -name ".tmp" -exec rm -rf "{}" +

deepclean: clean
	@echo "Removing package binaries..."
	@find ./ -type d -name "build" -exec rm -rf "{}" +
	@find ./ -type d -name "*.egg-info" -exec rm -rf "{}" +
	@find ./ -type d -name "__pycache__" -exec rm -rf "{}" +
	@echo "Removing virtual environment..."
	@find ./ -type d -name ".venv" -exec rm -rf "{}" +
