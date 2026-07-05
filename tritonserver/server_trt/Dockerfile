##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

# Includes both the TensorRT and Python backends used by the ensemble.
FROM nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /opt/mobileclip

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "numpy>=1.24"
RUN pip uninstall -y pillow || true
RUN CC="cc -mavx2" pip install --no-cache-dir --force-reinstall --no-deps \
    "pillow-simd>=9.5.0.post2"
