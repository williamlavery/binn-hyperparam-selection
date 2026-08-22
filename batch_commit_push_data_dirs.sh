#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TARGET_DIRS=(
  "Training/binn/dataX1num_11"
  "Training/binn/dataX1num_38"
)

TOTAL_BATCHES=20
COMMIT_PREFIX="${COMMIT_PREFIX:-Add batch of BINN models}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
BRANCH_NAME="${BRANCH_NAME:-$(git branch --show-current)}"
LFS_THRESHOLD_MB="${LFS_THRESHOLD_MB:-90}"
LFS_THRESHOLD_BYTES=$(( LFS_THRESHOLD_MB * 1024 * 1024 ))

get_file_size_bytes() {
  local file="$1"

  if stat -f '%z' "$file" >/dev/null 2>&1; then
    stat -f '%z' "$file"
  else
    stat -c '%s' "$file"
  fi
}

if [[ -z "$BRANCH_NAME" ]]; then
  echo "Unable to determine the current git branch." >&2
  exit 1
fi

for dir in "${TARGET_DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "Missing target directory: $dir" >&2
    exit 1
  fi
done

if ! git lfs version >/dev/null 2>&1; then
  echo "git-lfs is required but was not found in PATH." >&2
  exit 1
fi

last_completed_batch=""
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  last_completed_batch="$(
    git log --format=%s --grep="^${COMMIT_PREFIX} [0-9]+/${TOTAL_BATCHES}$" \
      | sed -nE "s/^${COMMIT_PREFIX} ([0-9]+)\/${TOTAL_BATCHES}$/\1/p" \
      | sort -n \
      | tail -n 1
  )"
fi

if [[ -z "$last_completed_batch" ]]; then
  last_completed_batch=0
fi

if (( last_completed_batch >= TOTAL_BATCHES )); then
  echo "All $TOTAL_BATCHES batches already appear to be completed on branch '$BRANCH_NAME'."
  exit 0
fi

next_batch_number=$(( last_completed_batch + 1 ))
remaining_batches=$(( TOTAL_BATCHES - last_completed_batch ))

FILES=()
while IFS= read -r file; do
  FILES+=( "$file" )
done < <(
  git ls-files --others --exclude-standard -- "${TARGET_DIRS[@]}" | LC_ALL=C sort
)

TOTAL_FILES="${#FILES[@]}"

if (( TOTAL_FILES == 0 )); then
  echo "No untracked files found under the target directories." >&2
  exit 1
fi

if (( TOTAL_FILES < remaining_batches )); then
  echo "Found only $TOTAL_FILES untracked files, fewer than the $remaining_batches remaining batches." >&2
  exit 1
fi

LFS_FILES=()
for file in "${FILES[@]}"; do
  if (( $(get_file_size_bytes "$file") > LFS_THRESHOLD_BYTES )); then
    LFS_FILES+=( "$file" )
  fi
done

if (( ${#LFS_FILES[@]} > 0 )); then
  echo "Tracking ${#LFS_FILES[@]} files above ${LFS_THRESHOLD_MB} MB with Git LFS."
  for file in "${LFS_FILES[@]}"; do
    git lfs track "$file" >/dev/null
  done
fi

repo_meta_dirty=0
if [[ -n "$(git status --porcelain -- .gitattributes .gitignore)" ]]; then
  repo_meta_dirty=1
fi

base_batch_size=$(( TOTAL_FILES / remaining_batches ))
remainder=$(( TOTAL_FILES % remaining_batches ))
start_index=0

echo "Resuming at batch $next_batch_number/$TOTAL_BATCHES on branch '$BRANCH_NAME'."
echo "Preparing $remaining_batches sequential commits from $TOTAL_FILES untracked files."

for (( batch_number=next_batch_number; batch_number<=TOTAL_BATCHES; batch_number++ )); do
  batch_size="$base_batch_size"
  relative_batch_number=$(( batch_number - next_batch_number + 1 ))
  if (( relative_batch_number <= remainder )); then
    batch_size=$(( batch_size + 1 ))
  fi

  batch_files=( "${FILES[@]:start_index:batch_size}" )

  if (( ${#batch_files[@]} == 0 )); then
    echo "Batch $batch_number resolved to zero files; aborting." >&2
    exit 1
  fi

  echo "Batch $batch_number/$TOTAL_BATCHES: staging ${#batch_files[@]} files."
  git add -- "${batch_files[@]}"
  if (( repo_meta_dirty == 1 )); then
    git add -- .gitattributes .gitignore
    repo_meta_dirty=0
  fi

  commit_message="$COMMIT_PREFIX $batch_number/$TOTAL_BATCHES"
  git commit -m "$commit_message"
  git push "$REMOTE_NAME" "$BRANCH_NAME"

  start_index=$(( start_index + batch_size ))
done

echo "Completed $TOTAL_BATCHES add/commit/push cycles."
