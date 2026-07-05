##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

# Includes the TensorRT, DALI, and Python backends used by the ensemble.
# Image preprocessing runs on GPU via the bundled DALI backend, so no PIL/
# libjpeg toolchain is needed here -- the BLS model only needs numpy.
FROM nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /opt/mobileclip

RUN pip install --no-cache-dir \
    "numpy>=1.24"
