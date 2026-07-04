#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##
# Downloads the four MobileCLIP checkpoints used by the Triton backends
# into ./checkpoints/, which is mounted as /checkpoints inside the container.

set -euo pipefail

mkdir -p checkpoints

for model in S0; do
  lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
  echo "Downloading MobileCLIP-$model → checkpoints/mobileclip_${lower}.pt"
  uv run --with huggingface_hub hf download \
    "apple/MobileCLIP-$model" \
    "mobileclip_${lower}.pt" \
    --local-dir ./checkpoints
done

echo "Done. Checkpoints:"
ls -lh checkpoints/
