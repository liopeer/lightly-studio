##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025-2026 Lionel Peer
##

MODEL_NAME := mobileclip_s0
CHECKPOINT := checkpoints/$(MODEL_NAME).pt
ENGINE_DIR := model_repository/mobileclip_s0_image_backend_trt/1
PLAN := $(ENGINE_DIR)/model.plan
ONNX := $(ENGINE_DIR)/model.onnx

# Preprocessing runs as a DALI pipeline serialized to model.dali. It's built
# by running the same tritonserver image the server itself uses (rather than
# adding nvidia-dali as a shared_deps_server dependency), so the DALI version
# that serializes the pipeline always matches the one that later loads it.
TRITON_IMAGE := nvcr.io/nvidia/tritonserver:25.03-py3
DALI_DIR := model_repository/mobileclip_s0_image_preprocessing/1
DALI_PIPELINE := $(DALI_DIR)/model.dali
DALI_MAX_BATCH_SIZE := 64

.PHONY: download plan dali up down logs clean

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
		--opt-batch-size 256 \
		--max-batch-size 1024 \
		--workspace-size-gib 8 \
		--normalize-embeddings \
		--out $(PLAN) \
		--onnx-out $(ONNX)

$(CHECKPOINT):
	$(MAKE) download

plan: $(PLAN)

$(DALI_PIPELINE): dali_pipeline.py
	mkdir -p $(DALI_DIR)
	docker run --rm --gpus all \
		--user $(shell id -u):$(shell id -g) \
		-v $(CURDIR):/workspace \
		-w /workspace \
		$(TRITON_IMAGE) \
		python3 dali_pipeline.py --out $(DALI_PIPELINE) --max-batch-size $(DALI_MAX_BATCH_SIZE)

dali: $(DALI_PIPELINE)

up: $(PLAN) $(DALI_PIPELINE)
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f triton-server-trt

clean:
	rm -f $(PLAN) $(ONNX) $(DALI_PIPELINE)
