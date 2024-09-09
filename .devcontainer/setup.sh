#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "/workspaces/${PWD##*/}/.venv" ]; then
    python -m venv /workspaces/${PWD##*/}/.venv
fi

# Activate virtual environment
source /workspaces/${PWD##*/}/.venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Add venv activation to bash startup script
echo "source /workspaces/${PWD##*/}/.venv/bin/activate" >> ~/.bashrc
