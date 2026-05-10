"""Qwen2.5-VL embedding generator.

Images: visual encoder patch tokens are mean-pooled → fixed-size vector.
Text:   language backbone last hidden state is mean-pooled → same size vector.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import UUID

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from lightly_studio.dataset.embedding_generator import ImageEmbeddingGenerator
from lightly_studio.models.embedding_model import EmbeddingModelCreate

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

logger = logging.getLogger(__name__)


class Qwen25VLEmbeddingGenerator(ImageEmbeddingGenerator):
    """Qwen2.5-VL image and text embedding generator.

    Image embeddings are produced by the visual encoder. Each image is passed
    through the vision transformer and the resulting patch tokens are mean-pooled
    into a single vector of size ``hidden_size``.

    Text embeddings are produced by the language model backbone. The input tokens
    are run through the decoder and the last hidden state is mean-pooled.

    Both embeddings share the same dimension (``model.config.hidden_size``), which
    is 3584 for the 7B variant and 1536 for the 2B variant.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Initialize the Qwen2.5-VL embedding model.

        Args:
            model_name: HuggingFace model identifier (e.g. "Qwen/Qwen2.5-VL-7B-Instruct").
        """
        self._model_name = model_name

        # Auto select device: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
            torch_dtype = torch.bfloat16
        else:
            self._device = torch.device("cpu")
            torch_dtype = torch.float32

        logger.info(
            "Loading %s on %s with dtype %s ...", model_name, self._device, torch_dtype
        )

        if self._device.type == "cuda":
            # device_map="auto" distributes across all available GPUs automatically.
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            ).to(self._device)

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_name)

        # Embedding dimension matches the LLM hidden size (e.g. 3584 for 7B).
        self._embedding_dimension: int = self._model.config.hidden_size
        self._model_hash = self._compute_model_hash()

        logger.info(
            "Loaded %s — embedding dimension: %d", model_name, self._embedding_dimension
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_model_hash(self) -> str:
        """Fingerprint: model name + first 10 parameter tensors (fast)."""
        h = hashlib.sha256()
        h.update(self._model_name.encode())
        for name, param in list(self._model.named_parameters())[:10]:
            h.update(name.encode())
            data = param.data.cpu().to(torch.float32).flatten()[:64].numpy()
            h.update(data.tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # EmbeddingGenerator protocol
    # ------------------------------------------------------------------

    def get_embedding_model_input(self, collection_id: UUID) -> EmbeddingModelCreate:
        """Return metadata needed to register this model in LightlyStudio.

        Args:
            collection_id: The ID of the collection.

        Returns:
            An EmbeddingModelCreate instance describing this model.
        """
        return EmbeddingModelCreate(
            name=self._model_name,
            embedding_model_hash=self._model_hash,
            embedding_dimension=self._embedding_dimension,
            collection_id=collection_id,
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed text using the language backbone (mean-pooled last hidden state).

        Args:
            text: The text to embed.

        Returns:
            A list of floats of length ``embedding_dimension``.
        """
        enc = self._processor.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            # model.model is the Qwen2_5_VLModel decoder (without the LM head).
            outputs = self._model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
            # Masked mean pooling so padding tokens don't contribute.
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_size)
            embedding = embedding.squeeze(0)  # (hidden_size,)

        return embedding.cpu().to(torch.float32).numpy().tolist()

    # ------------------------------------------------------------------
    # ImageEmbeddingGenerator protocol
    # ------------------------------------------------------------------

    def embed_images(
        self, filepaths: list[str], show_progress: bool = True
    ) -> NDArray[np.float32]:
        """Embed images using the visual encoder (mean-pooled patch tokens).

        Each image is processed independently because Qwen2.5-VL uses dynamic
        resolution (variable number of patch tokens per image), making batching
        non-trivial. The patch tokens produced by the vision transformer are
        mean-pooled into a single vector.

        Args:
            filepaths: Paths to the images to embed.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Float32 array of shape ``(len(filepaths), embedding_dimension)``.
        """
        total = len(filepaths)
        if not total:
            return np.empty((0, self._embedding_dimension), dtype=np.float32)

        embeddings = np.empty((total, self._embedding_dimension), dtype=np.float32)

        with (
            tqdm(
                total=total,
                desc="Generating embeddings",
                unit=" images",
                disable=not show_progress,
            ) as pbar,
            torch.no_grad(),
        ):
            for i, filepath in enumerate(filepaths):
                with Image.open(filepath) as img:
                    image = img.convert("RGB")

                # The image processor returns `pixel_values` (preprocessed patches)
                # and `image_grid_thw` (temporal/height/width grid dimensions).
                image_inputs = self._processor.image_processor(
                    images=[image], return_tensors="pt"
                )
                pixel_values = image_inputs["pixel_values"].to(self._device)
                image_grid_thw = image_inputs["image_grid_thw"].to(self._device)

                # get_image_features runs the visual encoder + patch merger.
                # pooler_output is a tuple of (num_merged_patches, hidden_size)
                # tensors, one per image. We process one image at a time.
                outputs = self._model.get_image_features(
                    pixel_values, image_grid_thw=image_grid_thw
                )
                patch_tokens = outputs.pooler_output[0]  # (num_patches, hidden_size)

                # Mean pool over spatial patch tokens → one vector per image.
                embedding = patch_tokens.mean(dim=0)  # (hidden_size,)
                embeddings[i] = embedding.cpu().to(torch.float32).numpy()
                pbar.update(1)

        return embeddings


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Use the smaller 3B model for a faster test; swap for 7B in production.
    TEST_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Loading {TEST_MODEL} …")
    gen = Qwen25VLEmbeddingGenerator(model_name=TEST_MODEL)
    dim = gen._embedding_dimension
    print(f"  embedding dimension: {dim}")

    # Text embedding
    text_emb = gen.embed_text("A chest X-ray showing pneumonia.")
    assert len(text_emb) == dim, f"Expected {dim}, got {len(text_emb)}"
    print(f"  text embedding OK  — shape: ({len(text_emb)},)  norm: {np.linalg.norm(text_emb):.4f}")

    # Image embeddings — use any .jpg/.png files passed as CLI args, or skip.
    image_paths = [p for p in sys.argv[1:] if Path(p).exists()]
    if image_paths:
        img_embs = gen.embed_images(image_paths)
        assert img_embs.shape == (len(image_paths), dim)
        print(f"  image embeddings OK — shape: {img_embs.shape}  norm[0]: {np.linalg.norm(img_embs[0]):.4f}")
    else:
        print("  (no image paths provided — skipping image embedding test)")

    print("All checks passed.")
