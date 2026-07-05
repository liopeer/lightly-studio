##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

MODEL_NAME := mobileclip_s0
CHECKPOINT := checkpoints/$(MODEL_NAME).pt
ENGINE_DIR := model_repository/mobileclip_s0_image_backend_trt/1
PLAN := $(ENGINE_DIR)/model.plan
ONNX := $(ENGINE_DIR)/model.onnx

.PHONY: download plan up down logs clean

download:
	bash download_models.sh

$(PLAN): $(CHECKPOINT)
	uv run \
		--project ../server/shared_deps_server \
		--extra tensorrt \
		mobileclip-export-tensorrt \
		--encoder image \
		--model-name $(MODEL_NAME) \
		--checkpoint-path $(CHECKPOINT) \
		--precision fp16 \
		--min-batch-size 1 \
		--opt-batch-size 64 \
		--max-batch-size 256 \
		--normalize-embeddings \
		--out $(PLAN) \
		--onnx-out $(ONNX)

$(CHECKPOINT):
	$(MAKE) download

plan: $(PLAN)

up: $(PLAN)
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f triton-server-trt

clean:
	rm -f $(PLAN) $(ONNX)
