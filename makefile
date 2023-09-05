.PHONY: all build check clean
.DEFAULT_GOAL: all


# --- Default target ---
all: build check


# --- Package compilation & testing ---
build:
	pip install .
	pip install .[test]
	pip install .[workflow]

check:
	pytest -v


# --- Cleaning ---
clean:
	@echo "Removing log files..."
	@find ./ -type f -name "*.log" -delete
	@echo "Removing temp directory..."
	@find ./ -type d -name ".tmp" -exec rm -rf "{}" +
