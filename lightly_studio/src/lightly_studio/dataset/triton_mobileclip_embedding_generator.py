"""Triton-backed MobileCLIP embedding generator."""

from __future__ import annotations

from io import BytesIO
from uuid import UUID

import numpy as np
import tritonclient.grpc as grpcclient
import xxhash
from numpy.typing import NDArray
from PIL import Image

from lightly_studio.dataset import env
from lightly_studio.models.embedding_model import EmbeddingModelCreate

from .embedding_generator import ImageCrop, ImageEmbeddingGenerator
from .image_embedding import ImageEmbeddingResult

DEFAULT_MODEL_NAME = "mobileclip_s0"
SUPPORTED_MODEL_NAMES = {DEFAULT_MODEL_NAME}
EMBEDDING_DIMENSION = 512

_IMAGE_PATH_INPUT = "IMAGE_PATH"
_IMAGE_BYTES_INPUT = "IMAGE_BYTES"
_TEXT_INPUT = "TEXT"
_CROP_X_INPUT = "CROP_X"
_CROP_Y_INPUT = "CROP_Y"
_CROP_WIDTH_INPUT = "CROP_WIDTH"
_CROP_HEIGHT_INPUT = "CROP_HEIGHT"
_EMBEDDING_OUTPUT = "EMBEDDING"


class TritonMobileCLIPEmbeddingGenerator(ImageEmbeddingGenerator):
    """MobileCLIP embedding model served by Triton gRPC."""

    def __init__(self) -> None:
        """Initialize the Triton MobileCLIP embedding client.

        The Triton server URL and MobileCLIP variant are read from environment
        variables when the generator is created.

        Raises:
            ValueError: If the Triton server URL is not configured.
        """
        server_url = env.env.str("LIGHTLY_STUDIO_TRITON_MOBILECLIP_URL", default=None)
        if server_url is None:
            raise ValueError("LIGHTLY_STUDIO_TRITON_MOBILECLIP_URL must be set.")

        self._url = server_url
        self._model_name = env.env.str(
            "LIGHTLY_STUDIO_TRITON_MOBILECLIP_VARIANT", DEFAULT_MODEL_NAME
        )
        if self._model_name not in SUPPORTED_MODEL_NAMES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
            raise ValueError(
                f"Unsupported Triton MobileCLIP variant '{self._model_name}'. "
                f"Supported variants: {supported}."
            )
        self._client = grpcclient.InferenceServerClient(url=self._url)
        self._model_hash = _get_model_hash(url=self._url, model_name=self._model_name)

    def get_embedding_model_input(self, collection_id: UUID) -> EmbeddingModelCreate:
        """Generate an EmbeddingModelCreate instance.

        Args:
            collection_id: The ID of the collection.

        Returns:
            An EmbeddingModelCreate instance with the model details.
        """
        return EmbeddingModelCreate(
            name=self._model_name,
            embedding_model_hash=self._model_hash,
            embedding_dimension=EMBEDDING_DIMENSION,
            collection_id=collection_id,
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed text with Triton MobileCLIP.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the generated embedding.
        """
        inputs = [
            _bytes_input(name=_TEXT_INPUT, values=[text]),
        ]
        embeddings = self._infer_embeddings(inputs=inputs, expected_count=1)
        result: list[float] = embeddings[0].tolist()
        return result

    def embed_images(
        self, filepaths: list[str], show_progress: bool = True
    ) -> ImageEmbeddingResult:
        """Embed image paths with Triton MobileCLIP.

        Args:
            filepaths: A list of file paths to the images to embed.
            show_progress: Whether to show a progress bar during embedding.

        Returns:
            An ``ImageEmbeddingResult`` with embeddings in the same order as the
            input file paths.
        """
        _ = show_progress
        if not filepaths:
            return ImageEmbeddingResult(
                embeddings=np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32),
                kept_indices=[],
            )

        inputs = [
            _bytes_input(name=_IMAGE_PATH_INPUT, values=filepaths),
        ]
        embeddings = self._infer_embeddings(inputs=inputs, expected_count=len(filepaths))
        return ImageEmbeddingResult(
            embeddings=embeddings,
            kept_indices=list(range(len(filepaths))),
        )

    def embed_pil_images(
        self, images: list[Image.Image], show_progress: bool = True
    ) -> NDArray[np.float32]:
        """Embed in-memory PIL images with Triton MobileCLIP.

        Images are serialized losslessly as PNG files and sent to Triton, where
        the byte-image preprocessing pipeline decodes and normalizes them.

        Args:
            images: In-memory images to embed.
            show_progress: Whether to show a progress bar during embedding.

        Returns:
            A numpy array representing the generated embeddings in the same order
            as the input images.
        """
        _ = show_progress
        if not images:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        inputs = [_image_bytes_input(images=images)]
        return self._infer_embeddings(inputs=inputs, expected_count=len(images))

    def embed_image_crops(
        self, image_crops: list[ImageCrop], show_progress: bool = True
    ) -> NDArray[np.float32]:
        """Embed image crop definitions with Triton MobileCLIP.

        Args:
            image_crops: A list of image crop definitions to embed.
            show_progress: Whether to show a progress bar during embedding.

        Returns:
            A numpy array representing the generated embeddings in the same order
            as the input crops.
        """
        _ = show_progress
        if not image_crops:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        inputs = [
            _bytes_input(
                name=_IMAGE_PATH_INPUT,
                values=[image_crop.filepath for image_crop in image_crops],
            ),
            _int64_input(name=_CROP_X_INPUT, values=[image_crop.x for image_crop in image_crops]),
            _int64_input(name=_CROP_Y_INPUT, values=[image_crop.y for image_crop in image_crops]),
            _int64_input(
                name=_CROP_WIDTH_INPUT,
                values=[image_crop.width for image_crop in image_crops],
            ),
            _int64_input(
                name=_CROP_HEIGHT_INPUT,
                values=[image_crop.height for image_crop in image_crops],
            ),
        ]
        return self._infer_embeddings(inputs=inputs, expected_count=len(image_crops))

    def _infer_embeddings(
        self,
        inputs: list[grpcclient.InferInput],
        expected_count: int,
    ) -> NDArray[np.float32]:
        """Run Triton inference and validate the embedding output."""
        result = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput(_EMBEDDING_OUTPUT)],
        )
        output = result.as_numpy(_EMBEDDING_OUTPUT)
        if output is None:
            raise ValueError(f"Triton response is missing output '{_EMBEDDING_OUTPUT}'.")

        embeddings = np.asarray(output, dtype=np.float32)
        expected_shape = (expected_count, EMBEDDING_DIMENSION)
        if embeddings.shape != expected_shape:
            raise ValueError(
                "Triton response has invalid embedding shape "
                f"{embeddings.shape}; expected {expected_shape}."
            )
        return embeddings


def _bytes_input(name: str, values: list[str]) -> grpcclient.InferInput:
    encoded_values = [value.encode("utf-8") for value in values]
    infer_input = grpcclient.InferInput(name, [len(encoded_values)], "BYTES")
    infer_input.set_data_from_numpy(np.asarray(encoded_values, dtype=object))
    return infer_input


def _image_bytes_input(images: list[Image.Image]) -> grpcclient.InferInput:
    encoded_images = [_encode_image_as_png(image=image) for image in images]
    infer_input = grpcclient.InferInput(_IMAGE_BYTES_INPUT, [len(encoded_images)], "BYTES")
    infer_input.set_data_from_numpy(np.asarray(encoded_images, dtype=object))
    return infer_input


def _encode_image_as_png(image: Image.Image) -> bytes:
    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def _int64_input(name: str, values: list[int]) -> grpcclient.InferInput:
    infer_input = grpcclient.InferInput(name, [len(values)], "INT64")
    infer_input.set_data_from_numpy(np.asarray(values, dtype=np.int64))
    return infer_input


def _get_model_hash(url: str, model_name: str) -> str:
    hasher = xxhash.xxh64()
    hasher.update(f"triton-mobileclip-grpc:{url}:{model_name}")
    return hasher.hexdigest()
