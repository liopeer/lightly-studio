##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##
import io
import numpy as np
import pytest
import torch
from PIL import Image

from shared_deps import mobileclip
from shared_deps import MobileCLIPPreprocessor, MobileCLIPVariantConfig, MOBILECLIP_CONFIGS


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> np.ndarray:
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


class TestMobileCLIPConfigs:
    def test_all_variants_present(self):
        assert set(MOBILECLIP_CONFIGS.keys()) == {"mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"}

    def test_embed_dim(self):
        for cfg in MOBILECLIP_CONFIGS.values():
            assert cfg.embed_dim == 512

    def test_s0_image_size(self):
        assert MOBILECLIP_CONFIGS["mobileclip_s0"].image_size == 256

    def test_other_variants_image_size(self):
        for name in ("mobileclip_s1", "mobileclip_s2", "mobileclip_b"):
            assert MOBILECLIP_CONFIGS[name].image_size == 224


class TestMobileCLIPPreprocessor:
    @pytest.mark.parametrize("image_size", [224, 256])
    def test_output_shape(self, image_size: int):
        preprocessor = MobileCLIPPreprocessor(image_size=image_size)
        image_bytes = _make_jpeg_bytes()
        tensor = preprocessor(image_bytes)
        assert tensor.shape == (3, image_size, image_size)

    def test_output_dtype(self):
        preprocessor = MobileCLIPPreprocessor(image_size=224)
        tensor = preprocessor(_make_jpeg_bytes())
        assert tensor.dtype == torch.float32

    def test_output_range(self):
        preprocessor = MobileCLIPPreprocessor(image_size=224)
        tensor = preprocessor(_make_jpeg_bytes())
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0


class TestMobileCLIPImport:
    def test_create_model_and_transforms_importable(self):
        assert callable(mobileclip.create_model_and_transforms)

    def test_configs_loadable(self):
        for model_name in ("mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"):
            import json, os
            configs_dir = os.path.join(os.path.dirname(mobileclip.__file__), "configs")
            cfg = json.load(open(os.path.join(configs_dir, f"{model_name}.json")))
            assert "image_cfg" in cfg

    def test_no_open_clip_on_import(self):
        import sys
        assert "open_clip" not in sys.modules, "open_clip must not be imported at module load time"
