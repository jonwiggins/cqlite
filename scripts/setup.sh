#!/bin/sh
set -e

# Configure git to use the repo's hooks directory
git config core.hooksPath scripts
echo "Git hooks installed (core.hooksPath = scripts)"

# Ensure required components are installed
if ! rustup component list --installed | grep -q rustfmt; then
    echo "Installing rustfmt..."
    rustup component add rustfmt
fi

if ! rustup component list --installed | grep -q clippy; then
    echo "Installing clippy..."
    rustup component add clippy
fi

echo "Setup complete."
