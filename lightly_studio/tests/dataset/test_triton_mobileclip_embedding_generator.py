from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from lightly_studio.dataset.embedding_generator import ImageCrop
from lightly_studio.dataset.triton_mobileclip_embedding_generator import (
    EMBEDDING_DIMENSION,
    TritonMobileCLIPEmbeddingGenerator,
)


class FakeTritonResult:
    def __init__(self, output: NDArray[np.float32] | None) -> None:
        self.output = output

    def as_numpy(self, name: str) -> NDArray[np.float32] | None:
        assert name == "EMBEDDING"
        return self.output


class FakeTritonClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.calls: list[dict[str, Any]] = []

    def infer(
        self,
        model_name: str,
        inputs: list[Any],
        outputs: list[Any],
    ) -> FakeTritonResult:
        self.calls.append(
            {
                "model_name": model_name,
                "inputs": inputs,
                "outputs": outputs,
            }
        )
        batch_size = int(inputs[0]._get_tensor().shape[0])
        return FakeTritonResult(np.ones((batch_size, EMBEDDING_DIMENSION), dtype=np.float32))


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeTritonClient:
    created_client: FakeTritonClient | None = None

    def create_client(url: str) -> FakeTritonClient:
        nonlocal created_client
        created_client = FakeTritonClient(url=url)
        return created_client

    monkeypatch.setenv("LIGHTLY_STUDIO_TRITON_MOBILECLIP_URL", "localhost:8001")
    monkeypatch.delenv("LIGHTLY_STUDIO_TRITON_MOBILECLIP_VARIANT", raising=False)
    monkeypatch.setattr(
        "lightly_studio.dataset.triton_mobileclip_embedding_generator.grpcclient"
        ".InferenceServerClient",
        create_client,
    )

    generator = TritonMobileCLIPEmbeddingGenerator()
    assert created_client is not None
    cast(Any, generator)._client = created_client
    return created_client


def test_get_embedding_model_input(fake_client: FakeTritonClient) -> None:
    generator = TritonMobileCLIPEmbeddingGenerator()
    cast(Any, generator)._client = fake_client

    collection_id = uuid4()
    model_input = generator.get_embedding_model_input(collection_id=collection_id)

    assert model_input.name == "mobileclip_s0"
    assert model_input.embedding_dimension == EMBEDDING_DIMENSION
    assert model_input.collection_id == collection_id


def test_embed_images__sends_image_paths(fake_client: FakeTritonClient) -> None:
    generator = TritonMobileCLIPEmbeddingGenerator()
    cast(Any, generator)._client = fake_client

    embeddings = generator.embed_images(filepaths=["/data/a.jpg", "/data/b.jpg"])

    assert embeddings.shape == (2, EMBEDDING_DIMENSION)
    call = fake_client.calls[-1]
    assert call["model_name"] == "mobileclip_s0"
    input_tensors = cast(list[Any], call["inputs"])
    assert [tensor.name() for tensor in input_tensors] == ["IMAGE_PATH"]
    assert list(input_tensors[0]._get_tensor().shape) == [2]


def test_embed_pil_images__sends_png_bytes(fake_client: FakeTritonClient) -> None:
    generator = TritonMobileCLIPEmbeddingGenerator()
    cast(Any, generator)._client = fake_client
    images = [Image.new("RGB", (2, 3), color=(255, 0, 0))]

    embeddings = generator.embed_pil_images(images=images)

    assert embeddings.shape == (1, EMBEDDING_DIMENSION)
    input_tensors = cast(list[Any], fake_client.calls[-1]["inputs"])
    assert [tensor.name() for tensor in input_tensors] == ["IMAGE_BYTES"]
    assert input_tensors[0]._get_tensor().shape == [1]
    assert input_tensors[0]._raw_content[4:8] == b"\x89PNG"


def test_embed_pil_images__empty_input_returns_empty_array() -> None:
    generator = TritonMobileCLIPEmbeddingGenerator.__new__(TritonMobileCLIPEmbeddingGenerator)

    embeddings = generator.embed_pil_images(images=[])

    assert embeddings.shape == (0, EMBEDDING_DIMENSION)


def test_embed_text__sends_text(fake_client: FakeTritonClient) -> None:
    generator = TritonMobileCLIPEmbeddingGenerator()
    cast(Any, generator)._client = fake_client

    embedding = generator.embed_text(text="a cat")

    assert len(embedding) == EMBEDDING_DIMENSION
    input_tensors = cast(list[Any], fake_client.calls[-1]["inputs"])
    assert [tensor.name() for tensor in input_tensors] == ["TEXT"]
    assert list(input_tensors[0]._get_tensor().shape) == [1]


def test_embed_image_crops__sends_paths_and_crop_coordinates(
    fake_client: FakeTritonClient,
) -> None:
    generator = TritonMobileCLIPEmbeddingGenerator()
    cast(Any, generator)._client = fake_client
    image_crops = [
        ImageCrop(filepath="/data/a.jpg", x=1, y=2, width=3, height=4),
        ImageCrop(filepath="/data/b.jpg", x=5, y=6, width=7, height=8),
    ]

    embeddings = generator.embed_image_crops(image_crops=image_crops)

    assert embeddings.shape == (2, EMBEDDING_DIMENSION)
    input_tensors = cast(list[Any], fake_client.calls[-1]["inputs"])
    assert [tensor.name() for tensor in input_tensors] == [
        "IMAGE_PATH",
        "CROP_X",
        "CROP_Y",
        "CROP_WIDTH",
        "CROP_HEIGHT",
    ]
    assert [list(tensor._get_tensor().shape) for tensor in input_tensors] == [[2]] * 5


def test_embed_image_crops__empty_input_returns_empty_array() -> None:
    generator = TritonMobileCLIPEmbeddingGenerator.__new__(TritonMobileCLIPEmbeddingGenerator)

    embeddings = generator.embed_image_crops(image_crops=[])

    assert embeddings.shape == (0, EMBEDDING_DIMENSION)


def test_infer_embeddings__invalid_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadShapeClient(FakeTritonClient):
        def infer(
            self,
            model_name: str,
            inputs: list[Any],
            outputs: list[Any],
        ) -> FakeTritonResult:
            _ = model_name, inputs, outputs
            return FakeTritonResult(np.ones((1, EMBEDDING_DIMENSION + 1), dtype=np.float32))

    monkeypatch.setenv("LIGHTLY_STUDIO_TRITON_MOBILECLIP_URL", "localhost:8001")
    generator = TritonMobileCLIPEmbeddingGenerator()
    cast(Any, generator)._client = BadShapeClient(url="localhost:8001")

    with pytest.raises(ValueError, match="invalid embedding shape"):
        generator.embed_images(filepaths=["/data/a.jpg"])


def test_init__unsupported_variant_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTLY_STUDIO_TRITON_MOBILECLIP_URL", "localhost:8001")
    monkeypatch.setenv("LIGHTLY_STUDIO_TRITON_MOBILECLIP_VARIANT", "mobileclip_s1")

    with pytest.raises(ValueError, match="Unsupported Triton MobileCLIP variant"):
        TritonMobileCLIPEmbeddingGenerator()
