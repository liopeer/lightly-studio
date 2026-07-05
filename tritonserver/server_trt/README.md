# MobileCLIP-S0 TensorRT Triton Server

This is a secondary Triton setup for the MobileCLIP-S0 image encoder. It keeps
the existing `server/` setup unchanged and exposes an image-only `mobileclip_s0`
compatibility model.

Pipeline:

```text
IMAGE_PATH -> image preprocessing -> TensorRT FP16 image encoder -> L2 norm -> EMBEDDING
```

The text path is intentionally omitted.

## Generate the TensorRT Plan

Download the checkpoint:

```bash
cd server_trt
make download
```

Build the image engine in a CUDA/TensorRT environment:

```bash
make plan
```

The generated `model.plan` and intermediate `model.onnx` are ignored by git.

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
