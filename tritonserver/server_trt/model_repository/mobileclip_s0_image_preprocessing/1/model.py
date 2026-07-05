#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Lionel Peer
#
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

_IMAGE_SIZE = 256
_IMAGE_PATH_INPUT = "IMAGE_PATH"
_CROP_X_INPUT = "CROP_X"
_CROP_Y_INPUT = "CROP_Y"
_CROP_WIDTH_INPUT = "CROP_WIDTH"
_CROP_HEIGHT_INPUT = "CROP_HEIGHT"


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            image_path = _decode_scalar_string(
                pb_utils.get_input_tensor_by_name(request, _IMAGE_PATH_INPUT).as_numpy()
            )
            image_tensor = _preprocess_image(
                image_path,
                crop_box=_get_crop_box(request=request),
            )  # [3, H, W]
            out_tensor = pb_utils.Tensor("images", image_tensor)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses


def _get_crop_box(request):
    crop_inputs = [
        pb_utils.get_input_tensor_by_name(request, input_name)
        for input_name in (_CROP_X_INPUT, _CROP_Y_INPUT, _CROP_WIDTH_INPUT, _CROP_HEIGHT_INPUT)
    ]
    if all(crop_input is None for crop_input in crop_inputs):
        return None
    if any(crop_input is None for crop_input in crop_inputs):
        raise pb_utils.TritonModelException("All crop inputs must be provided together.")
    return tuple(_decode_scalar_int(crop_input.as_numpy()) for crop_input in crop_inputs)


def _decode_scalar_string(value):
    scalar = np.asarray(value).reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def _decode_scalar_int(value):
    return int(np.asarray(value).reshape(-1)[0])


def _preprocess_image(image_path, crop_box):
    image = Image.open(image_path).convert("RGB")
    if crop_box is not None:
        x, y, width, height = crop_box
        image = image.crop((x, y, x + width, y + height))

    image = _resize_shorter_side(image, _IMAGE_SIZE)
    image = _center_crop(image, _IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))


def _resize_shorter_side(image, size):
    width, height = image.size
    if width <= height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_width = int(size * width / height)
        new_height = size
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def _center_crop(image, size):
    width, height = image.size
    left = round((width - size) / 2)
    top = round((height - size) / 2)
    return image.crop((left, top, left + size, top + size))
