#!/bin/sh
# Configure git to use the repo's hooks directory.
git config core.hooksPath scripts
echo "Git hooks installed (core.hooksPath = scripts)"
