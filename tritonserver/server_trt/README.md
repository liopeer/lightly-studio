# MobileCLIP-S0 TensorRT Triton Server

This is a secondary Triton setup for the MobileCLIP-S0 image encoder. It keeps
the existing `server/` setup unchanged and exposes an image-only `mobileclip_s0`
compatibility model.

Pipeline:

```text
IMAGE_PATH -> GPU preprocessing (DALI: decode, crop, resize, normalize) -> TensorRT FP16 image encoder with L2 norm -> EMBEDDING
```

Image decode, cropping, resizing, and normalization all run on GPU via
Triton's built-in DALI backend (`dali_pipeline.py`) -- there is no CPU/PIL
preprocessing step.

The text path is intentionally omitted.

## Generate the TensorRT Plan and DALI Pipeline

Download the checkpoint:

```bash
cd server_trt
make download
```

Build the image engine and preprocessing pipeline in a CUDA environment:

```bash
make plan
make dali
```

The generated `model.plan`, `model.onnx`, and `model.dali` are ignored by
git.

## Run

```bash
cd server_trt
make up
```

The secondary server maps Triton ports to `8010` HTTP, `8011` gRPC, and `8012`
metrics so it can run alongside the original server.

The public model interface is:

- Model: `mobileclip_s0`
- Input: `IMAGE_PATH`
- Optional crop inputs: `CROP_X`, `CROP_Y`, `CROP_WIDTH`, `CROP_HEIGHT`
- Output: `EMBEDDING`

Image files must be visible inside the container at the same path passed to
`IMAGE_PATH`. Add read-only dataset mounts to `docker-compose.yml` as needed.
