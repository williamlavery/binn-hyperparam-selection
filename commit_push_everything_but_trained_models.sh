#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REMOTE_NAME="${REMOTE_NAME:-origin}"
BRANCH_NAME="${BRANCH_NAME:-$(git branch --show-current)}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-everything but trained models}"

EXCLUDE_DIRS=(
  "Training/binn/dataX1num_11"
  "Training/binn/dataX1num_38"
)

if [[ -z "$BRANCH_NAME" ]]; then
  echo "Unable to determine the current git branch." >&2
  exit 1
fi

for dir in "${EXCLUDE_DIRS[@]}"; do
  if [[ -e "$dir" ]]; then
    :
  else
    echo "Warning: excluded path not found: $dir" >&2
  fi
done

git add -A -- . \
  ":(exclude)${EXCLUDE_DIRS[0]}" \
  ":(exclude)${EXCLUDE_DIRS[1]}"

if git diff --cached --quiet; then
  echo "No changes staged outside the trained-model directories."
  exit 0
fi

git commit -m "$COMMIT_MESSAGE"
git push "$REMOTE_NAME" "$BRANCH_NAME"
