##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Generate by cloning https://github.com/triton-inference-server/server and
# running `python compose.py --backend tensorrt --backend python --backend onnxruntime --repoagent checksum --container-version 25.03`
FROM nvcr.io/nvidia/tritonserver:25.03-pyt-python-py3

WORKDIR /opt/mobileclip

COPY shared_deps_server/ /opt/mobileclip/shared_deps_server/
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    ./shared_deps_server/