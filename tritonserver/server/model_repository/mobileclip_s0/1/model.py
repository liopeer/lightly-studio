#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import numpy as np
import triton_python_backend_utils as pb_utils

_ZERO_EMBEDDING = np.zeros(512, dtype=np.float32)


class TritonPythonModel:
    def initialize(self, args):
        pass  # stateless; sub-models own their weights

    def execute(self, requests):
        responses = []
        for request in requests:
            image_input = pb_utils.get_input_tensor_by_name(request, "image_bytes")
            text_input = pb_utils.get_input_tensor_by_name(request, "text")

            image_emb = _ZERO_EMBEDDING.copy()
            text_emb = _ZERO_EMBEDDING.copy()

            if image_input is not None:
                # image_pipeline is max_batch_size 64: prepend batch dim [N] → [1, N]
                img_np = image_input.as_numpy()[np.newaxis, :]
                infer_req = pb_utils.InferenceRequest(
                    model_name="mobileclip_s0_image_pipeline",
                    requested_output_names=["embeddings"],
                    inputs=[pb_utils.Tensor("image_bytes", img_np)],
                )
                result = infer_req.exec()
                if result.has_error():
                    raise pb_utils.TritonModelException(result.error().message())
                image_emb = pb_utils.get_output_tensor_by_name(
                    result, "embeddings"
                ).as_numpy()[0]  # [1, 512] → [512]

            if text_input is not None:
                # text_pipeline is max_batch_size 64: prepend batch dim [1] → [1, 1]
                text_np = text_input.as_numpy()[np.newaxis, :]
                infer_req = pb_utils.InferenceRequest(
                    model_name="mobileclip_s0_text_pipeline",
                    requested_output_names=["embeddings"],
                    inputs=[pb_utils.Tensor("text", text_np)],
                )
                result = infer_req.exec()
                if result.has_error():
                    raise pb_utils.TritonModelException(result.error().message())
                text_emb = pb_utils.get_output_tensor_by_name(
                    result, "embeddings"
                ).as_numpy()[0]  # [1, 512] → [512]

            responses.append(
                pb_utils.InferenceResponse([
                    pb_utils.Tensor("image_embeddings", image_emb),
                    pb_utils.Tensor("text_embeddings", text_emb),
                ])
            )

        return responses
