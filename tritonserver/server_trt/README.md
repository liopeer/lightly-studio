# MobileCLIP-S0 TensorRT Triton Server

This is a secondary Triton setup for the MobileCLIP-S0 image and text
encoders. It keeps the existing `server/` setup unchanged and exposes a
`mobileclip_s0` compatibility model that accepts either input.

Pipelines:

```text
IMAGE_PATH  -> GPU preprocessing (DALI: decode, crop, resize, normalize) -> TensorRT FP16 image encoder with L2 norm -> EMBEDDING
IMAGE_BYTES -> GPU preprocessing (DALI: decode, resize, normalize)       -> TensorRT FP16 image encoder with L2 norm -> EMBEDDING
TEXT        -> CPU tokenization (Python backend)                          -> TensorRT FP16 text encoder with L2 norm  -> EMBEDDING
```

Image decode, cropping, resizing, and normalization all run on GPU via
Triton's built-in DALI backend (`dali_pipeline.py`) -- there is no CPU/PIL
preprocessing step. Text tokenization has no DALI equivalent, so it runs as a
lightweight CPU Python-backend step before the TensorRT text encoder.

## Generate the TensorRT Plans and DALI Pipeline

Download the checkpoint:

```bash
cd server_trt
make download
```

Build the image and text engines and the image preprocessing pipeline in a
CUDA environment:

```bash
make plan
make dali
```

The generated `model.plan`, `model.onnx`, and `model.dali` files are ignored
by git.

## Run

```bash
cd server_trt
make up
```

The secondary server maps Triton ports to `8010` HTTP, `8011` gRPC, and `8012`
metrics so it can run alongside the original server.

The public model interface is:

- Model: `mobileclip_s0`
- Input: exactly one of `IMAGE_PATH`, `IMAGE_BYTES`, or `TEXT`
- Optional crop inputs (with `IMAGE_PATH` only): `CROP_X`, `CROP_Y`, `CROP_WIDTH`, `CROP_HEIGHT`
- Output: `EMBEDDING`

Image files must be visible inside the container at the same path passed to
`IMAGE_PATH`. Add read-only dataset mounts to `docker-compose.yml` as needed.
`IMAGE_BYTES` accepts compressed image bytes and does not require a dataset mount.
