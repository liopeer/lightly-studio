#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##
# Downloads the MobileCLIP-S0 checkpoint used to build the TensorRT engine.

set -euo pipefail

mkdir -p checkpoints

echo "Downloading MobileCLIP-S0 -> checkpoints/mobileclip_s0.pt"
uv run --with huggingface_hub hf download \
  "apple/MobileCLIP-S0" \
  "mobileclip_s0.pt" \
  --local-dir ./checkpoints

echo "Done. Checkpoints:"
ls -lh checkpoints/
