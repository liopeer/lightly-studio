##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

# Includes the TensorRT, DALI, and Python backends used by the ensemble.
# Image preprocessing runs on GPU via the bundled DALI backend, so no PIL/
# libjpeg toolchain is needed here -- the BLS model only needs numpy. Text
# preprocessing (tokenization) has no DALI equivalent and reuses
# shared_deps.mobileclip's tokenizer, which pulls in torch/torchvision/
# open-clip-torch/timm as a side effect of module import; shared_deps is
# installed with --no-deps to skip its unrelated pillow-simd dependency
# (only used by the image encoder module, which the text path never imports).
FROM nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /opt/mobileclip

COPY server/shared_deps_server/ /opt/mobileclip/shared_deps_server/
RUN pip install --no-cache-dir --no-deps ./shared_deps_server/ \
    && pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "numpy>=1.24" \
    "open-clip-torch>=2.20" \
    "timm>=0.9.5" \
    "torch>=2.5,<3"
