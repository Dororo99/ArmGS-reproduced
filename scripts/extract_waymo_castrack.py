#!/usr/bin/env python3
"""Extract one Waymo scene from the full CAStrack result JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from armgs.data import extract_castrack_scene_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one keyed Waymo scene from the full CAStrack JSON so "
            "training does not repeatedly parse the large multi-scene file."
        )
    )
    parser.add_argument("source", type=Path, help="full CAStrack result JSON")
    parser.add_argument("destination", type=Path, help="scene JSON to create")
    parser.add_argument(
        "--sequence",
        required=True,
        help=(
            "Waymo context name, with or without the segment- prefix and "
            "_with_camera_labels suffix"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace destination if it already exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = extract_castrack_scene_json(
        args.source,
        args.destination,
        sequence=args.sequence,
        overwrite=args.overwrite,
    )
    print(f"CAStrack scene: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
