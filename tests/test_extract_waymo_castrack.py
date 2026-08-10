from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import armgs.data as armgs_data


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "extract_waymo_castrack.py"
)
MODULE_NAME = "_armgs_extract_waymo_castrack_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
extract_cli = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = extract_cli
SPEC.loader.exec_module(extract_cli)


_SEQUENCE = "10448102132863604198_472_000_492_000"
_KEY = f"segment-{_SEQUENCE}_with_camera_labels"


def _full_payload() -> dict[str, object]:
    return {
        _KEY: {
            "0": {
                "obj_ids": [17],
                "name": ["Vehicle"],
                "boxes_lidar": [[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.0]],
            }
        },
        "segment-another_context_with_camera_labels": {
            "0": {"obj_ids": [], "name": [], "boxes_lidar": []}
        },
    }


def test_castrack_public_api_is_exported() -> None:
    assert armgs_data.CASTRACK_ACTOR_SOURCE == "castrack"
    assert armgs_data.extract_castrack_scene_json is not None
    assert armgs_data.load_castrack_actor_tracks is not None
    assert {
        "CASTRACK_ACTOR_SOURCE",
        "extract_castrack_scene_json",
        "load_castrack_actor_tracks",
    }.issubset(armgs_data.__all__)


def test_parser_accepts_source_destination_sequence_and_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full.json"
    destination = tmp_path / "scene.json"

    args = extract_cli.parse_args(
        [
            str(source),
            str(destination),
            "--sequence",
            _SEQUENCE,
            "--overwrite",
        ]
    )

    assert args.source == source
    assert args.destination == destination
    assert args.sequence == _SEQUENCE
    assert args.overwrite is True


def test_cli_extracts_only_the_requested_key(tmp_path: Path) -> None:
    source = tmp_path / "full.json"
    destination = tmp_path / "nested" / "scene.json"
    source.write_text(json.dumps(_full_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(source),
            str(destination),
            "--sequence",
            _SEQUENCE,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"CAStrack scene: {destination.resolve()}"
    assert set(json.loads(destination.read_text(encoding="utf-8"))) == {_KEY}
