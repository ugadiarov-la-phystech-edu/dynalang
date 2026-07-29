#!/usr/bin/env bash

# Periodically keep only the lexicographically newest episode files in every
# <experiment>/episodes directory under ROOT.
#
# Usage:
#   cleanup_episodes.sh [ROOT] [INTERVAL_SECONDS] [FILES_TO_KEEP]
#
# Example:
#   cleanup_episodes.sh /path/to/data 300 1000

set -u

ROOT="${1:-data}"
INTERVAL="${2:-300}"
FILES_TO_KEEP="${3:-1000}"

if [[ ! -d "$ROOT" ]]; then
  echo "Error: experiment root does not exist: $ROOT" >&2
  exit 1
fi

if [[ ! "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: interval must be a positive integer: $INTERVAL" >&2
  exit 1
fi

if [[ ! "$FILES_TO_KEEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: file count must be a positive integer: $FILES_TO_KEEP" >&2
  exit 1
fi

ROOT="$(cd "$ROOT" && pwd)"

cleanup() {
  local episodes_dir
  local found=0
  local total_files
  local files_to_delete

  while IFS= read -r -d '' episodes_dir; do
    found=1
    total_files="$(
      find "$episodes_dir" -mindepth 1 -maxdepth 1 -type f -printf '.' | wc -c
    )"
    files_to_delete=$((total_files - FILES_TO_KEEP))

    if (( files_to_delete <= 0 )); then
      continue
    fi

    printf '%s Cleaning %s: deleting %s, keeping %s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$episodes_dir" \
      "$files_to_delete" "$FILES_TO_KEEP"

    # Episode names sort chronologically in Dynalang's dataset format.
    find "$episodes_dir" -mindepth 1 -maxdepth 1 -type f -print0 \
      | sort -z \
      | head -z -n "$files_to_delete" \
      | xargs -0 -r rm -f --
  done < <(
    find "$ROOT" -mindepth 2 -maxdepth 2 -type d -name episodes -print0
  )

  if (( found == 0 )); then
    printf '%s No experiment episode directories found under %s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$ROOT"
  fi
}

echo "Watching $ROOT; keeping the newest $FILES_TO_KEEP files per episodes directory every ${INTERVAL}s"

while true; do
  cleanup
  sleep "$INTERVAL"
done
