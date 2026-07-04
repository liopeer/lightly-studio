#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import json

import torch
import triton_python_backend_utils as pb_utils

from shared_deps.mobileclip_image_encoder import MobileCLIPTorchBackend

MODEL_NAME = "mobileclip_s0"


class TritonPythonModel:
    def initialize(self, args):
        device_type = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device_type}:{device_id}"

        checkpoint_path = json.loads(args["model_config"])["parameters"][
            "checkpoint_path"
        ]["string_value"]

        self.model = MobileCLIPTorchBackend(
            model_name=MODEL_NAME,
            checkpoint_path=checkpoint_path,
        )
        self.model.to(self.device).eval()

    def execute(self, requests):
        batches, sizes = [], []
        for r in requests:
            t = torch.as_tensor(
                pb_utils.get_input_tensor_by_name(r, "images").as_numpy(),
                device=self.device,
            )  # [B, 3, H, W] or [3, H, W]
            if t.ndim == 3:
                t = t.unsqueeze(0)
            batches.append(t)
            sizes.append(t.shape[0])

        combined = torch.cat(batches, dim=0)  # [sum(B_i), 3, H, W]
        with torch.no_grad():
            all_embeddings = self.model(combined)  # [sum(B_i), 512]

        return [
            pb_utils.InferenceResponse([
                pb_utils.Tensor("embeddings", emb.cpu().numpy())
            ])
            for emb in torch.split(all_embeddings, sizes, dim=0)
        ]
