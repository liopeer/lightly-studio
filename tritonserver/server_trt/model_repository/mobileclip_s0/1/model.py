#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Lionel Peer
#
import numpy as np
import triton_python_backend_utils as pb_utils

_IMAGE_PATH_INPUT = "IMAGE_PATH"
_CROP_X_INPUT = "CROP_X"
_CROP_Y_INPUT = "CROP_Y"
_CROP_WIDTH_INPUT = "CROP_WIDTH"
_CROP_HEIGHT_INPUT = "CROP_HEIGHT"
_EMBEDDING_OUTPUT = "EMBEDDING"
_INTERNAL_EMBEDDING_OUTPUT = "embeddings"


class TritonPythonModel:
    def initialize(self, args):
        pass  # stateless; sub-models own preprocessing, inference, and normalization

    def execute(self, requests):
        responses = []
        for request in requests:
            image_input = pb_utils.get_input_tensor_by_name(request, _IMAGE_PATH_INPUT)
            if image_input is None:
                raise pb_utils.TritonModelException(f"Provide {_IMAGE_PATH_INPUT}.")

            embedding = _infer_image_embeddings(request=request, image_input=image_input)
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(_EMBEDDING_OUTPUT, embedding.astype(np.float32)),
                    ]
                )
            )

        return responses


def _infer_image_embeddings(request, image_input):
    image_paths = _as_column(image_input.as_numpy())
    inputs = [pb_utils.Tensor(_IMAGE_PATH_INPUT, image_paths)]
    present_crop_names = set()
    for input_name in (_CROP_X_INPUT, _CROP_Y_INPUT, _CROP_WIDTH_INPUT, _CROP_HEIGHT_INPUT):
        crop_input = pb_utils.get_input_tensor_by_name(request, input_name)
        if crop_input is not None:
            present_crop_names.add(input_name)
            inputs.append(pb_utils.Tensor(input_name, _as_column(crop_input.as_numpy())))

    _validate_crop_inputs(present_crop_names=present_crop_names)
    result = _infer_pipeline(
        model_name="mobileclip_s0_image_pipeline",
        inputs=inputs,
    )
    return pb_utils.get_output_tensor_by_name(result, _INTERNAL_EMBEDDING_OUTPUT).as_numpy()


def _infer_pipeline(model_name, inputs):
    infer_req = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=[_INTERNAL_EMBEDDING_OUTPUT],
        inputs=inputs,
    )
    result = infer_req.exec()
    if result.has_error():
        raise pb_utils.TritonModelException(result.error().message())
    return result


def _as_column(values):
    return np.asarray(values).reshape(-1, 1)


def _validate_crop_inputs(present_crop_names):
    crop_names = {_CROP_X_INPUT, _CROP_Y_INPUT, _CROP_WIDTH_INPUT, _CROP_HEIGHT_INPUT}
    if present_crop_names and present_crop_names != crop_names:
        missing = ", ".join(sorted(crop_names - present_crop_names))
        raise pb_utils.TritonModelException(f"Missing crop inputs: {missing}.")
