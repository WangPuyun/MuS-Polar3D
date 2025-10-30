#!/bin/bash

# Automated versioned Git commit script: automatically generates v1, v2, v3... for each commit
# Usage: ./git_push.sh "Description of changes made"

# Check if commit message is provided
if [ -z "$1" ]; then
  echo "❌ Error: Please provide a commit message, e.g.: ./git_push.sh \"Updated training set\""
  exit 1
fi

# Ensure we're on the main branch
git checkout main

# Automatically read and update version number
VERSION_FILE="version.txt"
if [ ! -f "$VERSION_FILE" ]; then
  echo "v1" > "$VERSION_FILE"
  VERSION="v1"
else
  LAST_VERSION=$(cat $VERSION_FILE)
  NUM=${LAST_VERSION#v}
  NEW_NUM=$((NUM + 1))
  VERSION="v$NEW_NUM"
  echo "$VERSION" > "$VERSION_FILE"
fi

# Add version.txt to commit
git add -A

# Construct complete commit message
COMMIT_MSG="$VERSION: $1"
git commit -m "$COMMIT_MSG"

# Push to remote repository
git push origin main

echo "✅ [$VERSION] Successfully committed and pushed:$1"
