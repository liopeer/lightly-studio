#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        device_type = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        device_id = args["model_instance_device_id"]
        self.device = f"{device_type}:{device_id}"

    def execute(self, requests):
        batch = torch.stack([
            torch.from_numpy(
                pb_utils.get_input_tensor_by_name(r, "raw_embeddings").as_numpy()
            )
            for r in requests
        ]).to(self.device)  # [B, 512]
        normalized = F.normalize(batch, dim=-1)  # [B, 512]
        return [
            pb_utils.InferenceResponse([
                pb_utils.Tensor("embeddings", normalized[i].cpu().numpy().astype(np.float32))
            ])
            for i in range(len(requests))
        ]
