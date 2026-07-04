##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tritonclient[http]",
#   "numpy",
# ]
# ///

import argparse
import sys

import numpy as np
import tritonclient.http as httpclient

MODELS = ["mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b"]

SAMPLE_TEXTS = [
    "a photo of a dog",
    "a cat sitting on a chair",
    "a beautiful mountain landscape",
    "people walking in a city street",
    "a red car parked outside",
    "sunset over the ocean",
    "a close-up of a flower",
    "children playing in a park",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MobileCLIP text inference via Triton.")
    parser.add_argument(
        "--texts",
        nargs="+",
        default=None,
        metavar="TEXT",
        help="Texts to embed. Defaults to built-in sample texts.",
    )
    parser.add_argument("--model", default="mobileclip_s0", choices=MODELS)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--output", default=None, help="Optional .npz path to save embeddings.")
    args = parser.parse_args()

    texts = args.texts if args.texts is not None else SAMPLE_TEXTS
    print(f"Encoding {len(texts)} texts. Model: {args.model}")

    client = httpclient.InferenceServerClient(url=f"{args.host}:{args.port}")
    results: dict[str, np.ndarray] = {}
    n_succeeded = 0

    for i, text in enumerate(texts, 1):
        text_data = np.array([text.encode()], dtype=object)
        txt_input = httpclient.InferInput("TEXT", [1], "BYTES")
        txt_input.set_data_from_numpy(text_data)

        try:
            result = client.infer(
                model_name=args.model,
                inputs=[txt_input],
                outputs=[httpclient.InferRequestedOutput("EMBEDDING")],
            )
            emb = result.as_numpy("EMBEDDING")[0]
            results[text] = emb
            n_succeeded += 1
            print(f"[{i:4d}/{len(texts)}] {text!r}  txt_norm={np.linalg.norm(emb):.4f}")
        except Exception as exc:
            print(f"[{i:4d}/{len(texts)}] {text!r}  ERROR: {exc}", file=sys.stderr)

    print(f"\nDone: {n_succeeded}/{len(texts)} succeeded.")

    if args.output and results:
        ordered = list(results)
        arr = np.stack([results[t] for t in ordered])
        np.savez(args.output, texts=np.array(ordered), embeddings=arr)
        print(f"Saved {arr.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
