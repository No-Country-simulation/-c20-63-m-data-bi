#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "/workspaces/${PWD##*/}/.venv" ]; then
    python -m venv /workspaces/${PWD##*/}/.venv
fi

# Activate virtual environment
if [ -n "$CODESPACES" ]; then
    source /workspaces/${PWD##*/}/.venv/bin/activate
else
    # Add venv activation to bash startup script for local development
    echo "source /workspaces/${PWD##*/}/.venv/bin/activate" >> ~/.bashrc
fi

# Install requirements
pip install -r requirements.txt
