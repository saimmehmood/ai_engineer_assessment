.PHONY: init run clean env help

# Set Python interpreter
PYTHON := python3
VENV := .venv
VENV_ACTIVATE := $(VENV)/bin/activate

# Initialize the project: create venv and install dependencies
init: env
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	. $(VENV_ACTIVATE) && pip install -r requirements.txt
	@echo "Setup complete. Activate your virtual environment with: source $(VENV_ACTIVATE)"

# Set up environment variables
env:
	@echo "Setting up environment variables..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from .env.example. Please update with your actual API keys."; \
	else \
		echo ".env file already exists. No changes made."; \
	fi

# Start the Flask server
run:
	@echo "Starting Flask server..."
	. $(VENV_ACTIVATE) && FLASK_APP=application.py python -m flask run

# Clean up virtual environment
clean:
	@echo "Cleaning up virtual environment..."
	rm -rf $(VENV)

# Help command that shows available commands
help:
	@echo "Available commands:"
	@echo "  make init  - Create virtual environment, install dependencies, and setup env file"
	@echo "  make env   - Create .env file from .env.example if it doesn't exist"
	@echo "  make start - Start the Flask server"
	@echo "  make clean - Remove virtual environment" 