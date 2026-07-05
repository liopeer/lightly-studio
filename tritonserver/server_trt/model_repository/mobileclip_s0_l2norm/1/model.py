#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Lionel Peer
#
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        batches = [
            pb_utils.get_input_tensor_by_name(request, "raw_embeddings")
            .as_numpy()
            .astype(np.float32, copy=False)
            for request in requests
        ]
        normalized_batches = [_l2_normalize(batch) for batch in batches]
        return [
            pb_utils.InferenceResponse([
                pb_utils.Tensor("embeddings", normalized_batch)
            ])
            for normalized_batch in normalized_batches
        ]


def _l2_normalize(batch):
    norms = np.linalg.norm(batch, axis=-1, keepdims=True)
    return np.divide(
        batch,
        np.maximum(norms, np.finfo(np.float32).eps),
        dtype=np.float32,
    )
