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
import asyncio
import sys
from pathlib import Path

import numpy as np
import tritonclient.grpc.aio as grpcclient

MODELS = ["mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


async def infer_image(
    client: grpcclient.InferenceServerClient,
    model: str,
    paths: list[Path],
    results: dict[Path, np.ndarray],
) -> bool:
    path_data = np.array([str(path).encode("utf-8") for path in paths], dtype=object)
    img_input = grpcclient.InferInput("IMAGE_PATH", [len(paths)], "BYTES")
    img_input.set_data_from_numpy(path_data)

    try:
        result = await client.infer(
            model_name=model,
            inputs=[img_input],
            outputs=[grpcclient.InferRequestedOutput("EMBEDDING")],
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False

    embeddings = result.as_numpy("EMBEDDING")
    if len(embeddings) != len(paths):
        print(
            f"ERROR: expected {len(paths)} embeddings, got {len(embeddings)}.",
            file=sys.stderr,
        )
        return False

    total = len(paths)
    for index, (path, emb) in enumerate(zip(paths, embeddings), 1):
        results[path] = emb
        print(f"[{index:4d}/{total}] {path.name}  img_norm={np.linalg.norm(emb):.4f}")
    return True


async def run(args: argparse.Namespace, images: list[Path]) -> dict[Path, np.ndarray]:
    results: dict[Path, np.ndarray] = {}

    async with grpcclient.InferenceServerClient(url=f"{args.host}:{args.port}") as client:
        succeeded = await infer_image(client, args.model, images, results)

    print(f"\nDone: {len(results) if succeeded else 0}/{len(images)} succeeded.")
    return results


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

    results = asyncio.run(run(args, images))

    if args.output and results:
        ordered = sorted(results)
        arr = np.stack([results[p] for p in ordered])
        np.savez(args.output, paths=np.array([str(p) for p in ordered]), embeddings=arr)
        print(f"Saved {arr.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
