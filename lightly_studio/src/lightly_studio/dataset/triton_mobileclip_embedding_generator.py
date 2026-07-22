"""Triton-backed path-based embedding generator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import numpy as np
import tritonclient.grpc as grpcclient
import xxhash
from numpy.typing import NDArray

from lightly_studio.models.embedding_model import EmbeddingModelCreate

from .embedding_generator import ImageCrop, ImageEmbeddingGenerator
from .image_embedding import ImageEmbeddingResult

# Maximum number of inputs sent in one Triton inference request.
DEFAULT_REQUEST_BATCH_SIZE = 512
_MIN_EMBEDDING_OUTPUT_RANK = 2
_IMAGE_PATH_INPUT = "IMAGE_PATH"
_TEXT_INPUT = "TEXT"
_CROP_X_INPUT = "CROP_X"
_CROP_Y_INPUT = "CROP_Y"
_CROP_WIDTH_INPUT = "CROP_WIDTH"
_CROP_HEIGHT_INPUT = "CROP_HEIGHT"
_EMBEDDING_OUTPUT = "EMBEDDING"


class TritonEmbeddingGenerator(ImageEmbeddingGenerator):
    """Embedding model served by a compatible Triton gRPC endpoint.

    The endpoint must accept ``TEXT`` or ``IMAGE_PATH`` inputs, optionally accept
    crop-coordinate inputs, and return an ``EMBEDDING`` output.
    """

    def __init__(self, url: str, model_name: str) -> None:
        """Initialize the Triton client and read the embedding output dimension."""
        self._url = url
        self._model_name = model_name
        self._client = grpcclient.InferenceServerClient(url=url)
        metadata = self._client.get_model_metadata(model_name=model_name)
        self._embedding_dimension = _get_embedding_dimension(metadata=metadata)
        self._model_hash = _get_model_hash(url=url, model_name=model_name)

    def get_embedding_model_input(self, collection_id: UUID) -> EmbeddingModelCreate:
        """Build the database representation of this embedding model."""
        return EmbeddingModelCreate(
            name=self._model_name,
            embedding_model_hash=self._model_hash,
            embedding_dimension=self._embedding_dimension,
            collection_id=collection_id,
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed a text query."""
        embeddings = self._infer_embeddings(
            inputs=[_bytes_input(name=_TEXT_INPUT, values=[text])], expected_count=1
        )
        result: list[float] = embeddings[0].tolist()
        return result

    def embed_images(
        self, filepaths: list[str], show_progress: bool = True
    ) -> ImageEmbeddingResult:
        """Embed image paths, forwarding remote URLs unchanged to Triton."""
        _ = show_progress
        embeddings = self._embed_in_chunks(
            values=filepaths,
            make_inputs=lambda paths: [_bytes_input(name=_IMAGE_PATH_INPUT, values=paths)],
        )
        return ImageEmbeddingResult(
            embeddings=embeddings,
            kept_indices=list(range(len(filepaths))),
        )

    def embed_image_crops(
        self, image_crops: list[ImageCrop], show_progress: bool = True
    ) -> NDArray[np.float32]:
        """Embed image crops represented by their source paths and coordinates."""
        _ = show_progress
        return self._embed_in_chunks(
            values=image_crops,
            make_inputs=_crop_inputs,
        )

    def _embed_in_chunks(
        self,
        values: Sequence[str] | Sequence[ImageCrop],
        make_inputs: Any,
    ) -> NDArray[np.float32]:
        if not values:
            return np.empty((0, self._embedding_dimension), dtype=np.float32)

        chunks = [
            self._infer_embeddings(
                inputs=make_inputs(values[index : index + DEFAULT_REQUEST_BATCH_SIZE]),
                expected_count=len(values[index : index + DEFAULT_REQUEST_BATCH_SIZE]),
            )
            for index in range(0, len(values), DEFAULT_REQUEST_BATCH_SIZE)
        ]
        return np.concatenate(chunks, axis=0)

    def _infer_embeddings(
        self, inputs: list[grpcclient.InferInput], expected_count: int
    ) -> NDArray[np.float32]:
        result = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput(_EMBEDDING_OUTPUT)],
        )
        output = result.as_numpy(_EMBEDDING_OUTPUT)
        if output is None:
            raise ValueError(f"Triton response is missing output '{_EMBEDDING_OUTPUT}'.")

        embeddings = np.asarray(output, dtype=np.float32)
        expected_shape = (expected_count, self._embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                "Triton response has invalid embedding shape "
                f"{embeddings.shape}; expected {expected_shape}."
            )
        return embeddings


def _bytes_input(name: str, values: Sequence[str]) -> grpcclient.InferInput:
    encoded_values = [value.encode("utf-8") for value in values]
    infer_input = grpcclient.InferInput(name, [len(encoded_values)], "BYTES")
    infer_input.set_data_from_numpy(np.asarray(encoded_values, dtype=object))
    return infer_input


def _crop_inputs(image_crops: Sequence[ImageCrop]) -> list[grpcclient.InferInput]:
    return [
        _bytes_input(name=_IMAGE_PATH_INPUT, values=[crop.filepath for crop in image_crops]),
        _int64_input(name=_CROP_X_INPUT, values=[crop.x for crop in image_crops]),
        _int64_input(name=_CROP_Y_INPUT, values=[crop.y for crop in image_crops]),
        _int64_input(name=_CROP_WIDTH_INPUT, values=[crop.width for crop in image_crops]),
        _int64_input(name=_CROP_HEIGHT_INPUT, values=[crop.height for crop in image_crops]),
    ]


def _int64_input(name: str, values: Sequence[int]) -> grpcclient.InferInput:
    infer_input = grpcclient.InferInput(name, [len(values)], "INT64")
    infer_input.set_data_from_numpy(np.asarray(values, dtype=np.int64))
    return infer_input


def _get_embedding_dimension(metadata: Any) -> int:
    outputs = metadata["outputs"] if isinstance(metadata, dict) else metadata.outputs
    for output in outputs:
        name = output["name"] if isinstance(output, dict) else output.name
        if name != _EMBEDDING_OUTPUT:
            continue
        shape = output["shape"] if isinstance(output, dict) else output.shape
        if len(shape) < _MIN_EMBEDDING_OUTPUT_RANK or shape[-1] <= 0:
            raise ValueError(f"Triton output '{_EMBEDDING_OUTPUT}' has invalid shape {shape}.")
        return int(shape[-1])
    raise ValueError(f"Triton model does not expose output '{_EMBEDDING_OUTPUT}'.")


def _get_model_hash(url: str, model_name: str) -> str:
    hasher = xxhash.xxh64()
    hasher.update(f"triton-grpc:{url}:{model_name}")
    return hasher.hexdigest()
