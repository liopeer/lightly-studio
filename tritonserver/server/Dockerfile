##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Generate by cloning https://github.com/triton-inference-server/server and
# running `python compose.py --backend tensorrt --backend python --backend onnxruntime --repoagent checksum --container-version 25.03`
FROM nvcr.io/nvidia/tritonserver:25.03-pyt-python-py3

WORKDIR /opt/mobileclip

COPY shared_deps_server/ /opt/mobileclip/shared_deps_server/
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    ./shared_deps_server/ \
    && pip uninstall -y pillow \
    && CC="cc -mavx2" pip install --no-cache-dir --force-reinstall --no-deps \
    "pillow-simd>=9.5.0.post2"
