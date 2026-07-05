##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tritonclient[grpc]",
#   "numpy",
# ]
# ///

import argparse
import sys
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

MODELS = ["mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MobileCLIP inference on a folder of images via Triton."
    )
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--model", default="mobileclip_s0", choices=MODELS)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory.", file=sys.stderr)
        sys.exit(1)

    images = sorted({p for ext in IMAGE_EXTENSIONS for p in args.folder.rglob(f"*{ext}")})
    if not images:
        print(f"No images found in {args.folder}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images. Model: {args.model}")

    client = grpcclient.InferenceServerClient(url=f"{args.host}:{args.port}")
    results: dict[Path, np.ndarray] = {}
    n_succeeded = 0

    for i, path in enumerate(images, 1):
        path_data = np.array([str(path).encode("utf-8")], dtype=object)
        img_input = grpcclient.InferInput("IMAGE_PATH", [1], "BYTES")
        img_input.set_data_from_numpy(path_data)

        try:
            result = client.infer(
                model_name=args.model,
                inputs=[img_input],
                outputs=[grpcclient.InferRequestedOutput("EMBEDDING")],
            )
            emb = result.as_numpy("EMBEDDING")[0]
            results[path] = emb
            n_succeeded += 1
            print(f"[{i:4d}/{len(images)}] {path.name}  img_norm={np.linalg.norm(emb):.4f}")
        except Exception as exc:
            print(f"[{i:4d}/{len(images)}] {path.name}  ERROR: {exc}", file=sys.stderr)

    print(f"\nDone: {n_succeeded}/{len(images)} succeeded.")

    if args.output and results:
        ordered = sorted(results)
        arr = np.stack([results[p] for p in ordered])
        np.savez(args.output, paths=np.array([str(p) for p in ordered]), embeddings=arr)
        print(f"Saved {arr.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
