#!/usr/bin/env bash
# Fetch the SO-101 robot model from Google DeepMind's mujoco_menagerie.
#
# The model is a third-party asset (Apache-2.0), so it is not vendored into this
# repo. We need exactly one folder out of it (robotstudio_so101, about 18MB), so
# this does a sparse checkout rather than cloning the full 2.2GB collection.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/arm-bridge/menagerie"

if [ -f "$DEST/robotstudio_so101/so101.xml" ]; then
  echo "[assets] SO-101 model already present at $DEST/robotstudio_so101"
  exit 0
fi

echo "[assets] fetching SO-101 model from mujoco_menagerie (sparse, ~18MB)..."
rm -rf "$DEST"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set robotstudio_so101

if [ -f "$DEST/robotstudio_so101/so101.xml" ]; then
  echo "[assets] done: $DEST/robotstudio_so101"
else
  echo "[assets] FAILED: so101.xml not found after checkout" >&2
  exit 1
fi
