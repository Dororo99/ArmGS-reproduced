"""Benchmark the reference hash-grid's visible/chunked training memory."""

from __future__ import annotations

import argparse
import json
import time

import torch

from armgs.encodings import HashGridEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--visible-count", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this VRAM benchmark requires CUDA")
    torch.cuda.set_device(device)
    encoder = HashGridEncoder(
        num_levels=8,
        features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        max_resolution=2048,
    ).to(device)

    for count in args.counts:
        if count <= 0:
            raise ValueError("counts must be positive")
        visible_count = (
            count if args.visible_count is None else min(args.visible_count, count)
        )
        positions = (
            torch.rand(count, 3, device=device, requires_grad=True) * 2.0 - 1.0
        )
        visible = torch.arange(visible_count, device=device)
        encoder.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        features = encoder(
            positions,
            visible_indices=visible,
            chunk_size=args.chunk_size,
        )
        features.square().mean().backward()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        result = {
            "total_gaussians": count,
            "visible_gaussians": visible_count,
            "chunk_size": args.chunk_size,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
            / (1024**2),
            "elapsed_ms": elapsed_ms,
        }
        print(json.dumps(result, sort_keys=True))
        del features, positions, visible
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
