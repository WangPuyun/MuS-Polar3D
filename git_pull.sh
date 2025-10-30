#!/bin/bash

# One-click script to pull latest code from GitHub
# Usage: ./git_pull.sh

# Ensure we're on the main branch
git checkout main

# Pull latest updates from remote main branch
git reset --hard
git pull origin main

echo "✅ Successfully pulled latest code from remote repository (main branch)"
