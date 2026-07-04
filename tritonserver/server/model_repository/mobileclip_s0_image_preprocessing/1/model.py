#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import numpy as np
import triton_python_backend_utils as pb_utils

from shared_deps.mobileclip_image_encoder import MOBILECLIP_CONFIGS, MobileCLIPPreprocessor

MODEL_NAME = "mobileclip_s0"


class TritonPythonModel:
    def initialize(self, args):
        self.preprocessor = MobileCLIPPreprocessor(MOBILECLIP_CONFIGS[MODEL_NAME].image_size)

    def execute(self, requests):
        responses = []
        for request in requests:
            image_bytes_np = pb_utils.get_input_tensor_by_name(
                request, "image_bytes"
            ).as_numpy()
            image_tensor = self.preprocessor(image_bytes_np)  # [3, H, W]
            out_tensor = pb_utils.Tensor("images", image_tensor.numpy().astype(np.float32))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
