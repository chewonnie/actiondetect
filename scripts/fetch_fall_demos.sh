#!/usr/bin/env bash
# Fetch the 3 URFD fall demo clips used by app/dashboard.py Panel 1d.
# License: CC BY-NC-SA 4.0 (Bogdan Kwolek & Michal Kepski, 2014)
# Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/datasets/fall/urfd/fall"
mkdir -p "$DEST"
BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"

for n in 01 02 03; do
  out="$DEST/fall-${n}-cam0.mp4"
  if [ -s "$out" ]; then
    echo "[skip] $out exists"
    continue
  fi
  echo "[get ] $out"
  curl -sSL --retry 3 -o "$out" "$BASE/fall-${n}-cam0.mp4"
done
echo "done. (~4 MB total)"
