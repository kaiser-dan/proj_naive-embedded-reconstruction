.PHONY: all install check clean
.DEFAULT_GOAL: all

POETRY=~/.local/bin/poetry

# --- Default target ---
all: install check


# --- Package compilation & testing ---
install: $(POETRY)
	$(POETRY) install

check:
	$(POETRY) run pytest -v

# --- Cleaning ---
clean:
	@echo "Removing log files..."
	@find ./ -type f -name "*.log" -delete
	@echo "Removing temp directory..."
	@find ./ -type d -name ".tmp" -exec rm -rf "{}" +
