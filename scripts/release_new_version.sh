#!/bin/bash
set -euxo pipefail

# Usage
if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v0.1.0rc1"
    exit 1
fi

VERSION_TAG="$1"
VERSION="${VERSION_TAG#v}"  # Remove 'v' prefix
BRANCH_NAME="bump-${VERSION_TAG}"

# Check uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Uncommitted changes found. Please commit or stash them first."
    exit 1
fi

# Switch to main and pull
echo "Switching to main branch..."
git checkout main
git pull origin main

# Create new branch
echo "Creating branch: $BRANCH_NAME"
git checkout -b "$BRANCH_NAME"

# Update version in pyproject.toml
echo "Updating version to $VERSION..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
else
    sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
fi

# Commit and push
git add pyproject.toml
git commit -m "Bump version to $VERSION"
git push -u origin "$BRANCH_NAME"

# Create PR if gh CLI is available
if command -v gh &> /dev/null; then
    echo "Creating pull request..."
    gh pr create --title "Release $VERSION_TAG" \
                 --body "Bump version to \`$VERSION\`" \
                 --base main --head "$BRANCH_NAME"
    echo "Done! PR created."
else
    echo "Done! Create PR manually at:"
    echo "https://github.com/flashinfer-ai/flashinfer-bench/compare/main...$BRANCH_NAME"
fi
