#!/usr/bin/env bash
# Creates a snippet for each .csv in DIR
# Requires authentication via 'glab auth login'
# Assumes you run this script from the project root

set -euo pipefail
shopt -s nullglob

DIR="data"
REPO="nlp-mnm/nlp-project"

for file in "$DIR"/*.csv; do
  base="$(basename "$file")"     # e.g. myfile.csv
  title="${base%.csv}"           # e.g. myfile

  glab snippet create \
    -t "$title" \
    -f "$base" \
    -R "$REPO" \
    -v "public"\
    "$file"
done
