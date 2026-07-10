"""Index a dataset with Qwen2.5-VL embeddings and launch the LightlyStudio GUI."""

import logging

import lightly_studio as ls

from qwen25vl_embedding_generator import Qwen25VLEmbeddingGenerator
from lightly_studio.dataset.embedding_manager import EmbeddingManagerProvider
from lightly_studio.resolvers import sample_embedding_resolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = ls.ImageDataset.create()

logger.info("Registering Qwen2.5-VL embedding model …")

embedding_manager = EmbeddingManagerProvider.get_embedding_manager()
qwen_model = embedding_manager.register_embedding_model(
    session=dataset.session,
    collection_id=dataset.dataset_id,
    embedding_generator=Qwen25VLEmbeddingGenerator(),
    set_as_default=True,
)

dataset.add_samples_from_coco(
    annotations_json="datasets/rsna_pneumonia/valid/_annotations-fixed.coco.json",
    images_path="datasets/rsna_pneumonia/valid",
    embed=True,
)

ls.start_gui()
