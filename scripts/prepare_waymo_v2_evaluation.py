#!/usr/bin/env python3
"""Prepare an auditable ArmGS Waymo-v2 evaluation split.

This first-stage adapter validates the seven Waymo v2 parquet components,
extracts the requested camera images at the paper resolution, and writes
separate reconstruction and novel-view metric manifests. LiDAR/actor
canonicalization remains a training-adapter concern and is deliberately not
silently approximated here.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


WAYMO_CAMERA_IDS = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}
WAYMO_REQUIRED_COMPONENTS = (
    "camera_image",
    "camera_calibration",
    "lidar",
    "lidar_pose",
    "lidar_box",
    "lidar_calibration",
    "vehicle_pose",
)
_IMAGE_COLUMN = "[CameraImageComponent].image"
_TIMESTAMP_COLUMN = "key.frame_timestamp_micros"
_CAMERA_COLUMN = "key.camera_name"
_METRIC_PROTOCOLS = {
    "psnr": "mean-per-image-rgb-mse-data-range-1",
    "ssim": "3dgs-gaussian-11x11-sigma-1.5-data-range-1",
    "lpips": "official-lpips-v0.1-minus-one-to-one",
    "lpips_net": "alex",
}
_COMPONENT_KEY_COLUMNS = {
    "camera_image": (_TIMESTAMP_COLUMN, _CAMERA_COLUMN),
    "camera_calibration": (_CAMERA_COLUMN,),
    "lidar": (_TIMESTAMP_COLUMN, "key.laser_name"),
    "lidar_pose": (_TIMESTAMP_COLUMN, "key.laser_name"),
    "lidar_box": (_TIMESTAMP_COLUMN, "key.laser_object_id"),
    "lidar_calibration": ("key.laser_name",),
    "vehicle_pose": (_TIMESTAMP_COLUMN,),
}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_camera_channels(value: str) -> tuple[str, ...]:
    channels = tuple(
        component.strip().upper()
        for component in value.split(",")
        if component.strip()
    )
    if not channels:
        raise argparse.ArgumentTypeError("at least one camera is required")
    unknown = set(channels) - set(WAYMO_CAMERA_IDS)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown Waymo camera(s): " + ", ".join(sorted(unknown))
        )
    if len(channels) != len(set(channels)):
        raise argparse.ArgumentTypeError("camera channels must be unique")
    return channels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Waymo v2 components and prepare ArmGS paper-protocol "
            "reconstruction/novel-view evaluation manifests."
        )
    )
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--parquet-dir", default="training")
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--cameras",
        type=parse_camera_channels,
        default=("FRONT",),
        help="comma-separated channels (paper reproduction default: FRONT)",
    )
    parser.add_argument("--start-frame", type=_non_negative_int, default=0)
    parser.add_argument(
        "--end-frame",
        type=_non_negative_int,
        help="inclusive source capture index (StreetGS range semantics)",
    )
    parser.add_argument("--test-every", type=_positive_int, default=4)
    parser.add_argument(
        "--first-test-position",
        type=_non_negative_int,
        default=4,
        help="relative first held-out capture (StreetGS/ArmGS reproduction: 4)",
    )
    parser.add_argument("--target-height", type=_positive_int, default=1066)
    parser.add_argument("--target-width", type=_positive_int, default=1600)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extract-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _component_paths(
    root: Path,
    *,
    parquet_dir: str,
    sequence: str,
) -> dict[str, Path]:
    if not parquet_dir or Path(parquet_dir).name != parquet_dir:
        raise ValueError("--parquet-dir must be one directory name")
    if not sequence or Path(sequence).name != sequence:
        raise ValueError("--sequence must be one non-empty context name")
    base = root / parquet_dir
    paths = {
        component: base / component / f"{sequence}.parquet"
        for component in WAYMO_REQUIRED_COMPONENTS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing Waymo v2 component parquet(s): " + ", ".join(missing)
        )
    return paths


def _read_parquet_rows(
    path: Path,
    columns: Sequence[str],
    *,
    filters: Any | None = None,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "Waymo preparation requires optional dependency 'pyarrow'"
        ) from error
    try:
        return parquet.read_table(
            path,
            columns=list(columns),
            filters=filters,
        ).to_pylist()
    except Exception as error:
        raise ValueError(f"failed to read Waymo parquet {path}: {error}") from error


def _validate_component_schemas(paths: Mapping[str, Path]) -> None:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "Waymo preparation requires optional dependency 'pyarrow'"
        ) from error
    for component, required_columns in _COMPONENT_KEY_COLUMNS.items():
        path = paths[component]
        try:
            names = set(parquet.read_schema(path).names)
        except Exception as error:
            raise ValueError(
                f"failed to read Waymo {component} schema {path}: {error}"
            ) from error
        missing = set(required_columns) - names
        if missing:
            raise ValueError(
                f"Waymo {component} parquet is missing required column(s): "
                + ", ".join(sorted(missing))
            )


def _calibrations(
    path: Path,
    *,
    requested_ids: set[int],
    target_size: tuple[int, int],
) -> dict[int, dict[str, Any]]:
    columns = (
        _CAMERA_COLUMN,
        "[CameraCalibrationComponent].intrinsic.f_u",
        "[CameraCalibrationComponent].intrinsic.f_v",
        "[CameraCalibrationComponent].intrinsic.c_u",
        "[CameraCalibrationComponent].intrinsic.c_v",
        "[CameraCalibrationComponent].width",
        "[CameraCalibrationComponent].height",
    )
    rows = _read_parquet_rows(path, columns)
    target_height, target_width = target_size
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        camera_id = int(row[_CAMERA_COLUMN])
        if camera_id not in requested_ids:
            continue
        if camera_id in result:
            raise ValueError(f"duplicate calibration for Waymo camera {camera_id}")
        source_width = int(row["[CameraCalibrationComponent].width"])
        source_height = int(row["[CameraCalibrationComponent].height"])
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"invalid calibration image size for camera {camera_id}")
        # StreetGaussians' Waymo preprocessing scales by the requested width,
        # does not upsample, floors both dimensions, and uses the same factor
        # for both intrinsic rows. FRONT 1920x1280 therefore becomes
        # 1600x1066 at the paper setting.
        scale = min(1.0, target_width / source_width)
        resized_width = int(source_width * scale)
        resized_height = int(source_height * scale)
        if camera_id == WAYMO_CAMERA_IDS["FRONT"] and (
            resized_height != target_height or resized_width != target_width
        ):
            raise ValueError(
                "FRONT target size is incompatible with StreetGS uniform "
                f"width scaling: requested {(target_height, target_width)}, "
                f"derived {(resized_height, resized_width)}"
            )
        result[camera_id] = {
            "source_size": [source_height, source_width],
            "target_size": [resized_height, resized_width],
            "uniform_scale": scale,
            "intrinsics": [
                [
                    float(row["[CameraCalibrationComponent].intrinsic.f_u"])
                    * scale,
                    0.0,
                    float(row["[CameraCalibrationComponent].intrinsic.c_u"])
                    * scale,
                ],
                [
                    0.0,
                    float(row["[CameraCalibrationComponent].intrinsic.f_v"])
                    * scale,
                    float(row["[CameraCalibrationComponent].intrinsic.c_v"])
                    * scale,
                ],
                [0.0, 0.0, 1.0],
            ],
        }
    missing = requested_ids - set(result)
    if missing:
        raise ValueError(
            "camera calibration is missing requested id(s): "
            + ", ".join(str(value) for value in sorted(missing))
        )
    return result


def _atomic_save_png(
    encoded_image: bytes,
    destination: Path,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Waymo image extraction requires optional dependency 'Pillow'"
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_height, source_width = source_size
    target_height, target_width = target_size
    try:
        with Image.open(BytesIO(encoded_image)) as image:
            image = image.convert("RGB")
            if image.size != (source_width, source_height):
                raise ValueError(
                    f"encoded image has size {image.size}, expected "
                    f"{(source_width, source_height)}"
                )
            if image.size != (target_width, target_height):
                image = image.resize(
                    (target_width, target_height),
                    Image.Resampling.BILINEAR,
                )
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                # The source Waymo payload is already JPEG-compressed. Store
                # the decoded (and possibly resized) RGB pixels losslessly so
                # preparation does not add a second lossy encoding to metric GT.
                image.save(handle, format="PNG", compress_level=3)
        os.replace(temporary, destination)
    except Exception:
        temporary_path = locals().get("temporary")
        if isinstance(temporary_path, Path):
            temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
    os.replace(temporary, path)


def _invalidate_generated_manifests(output_directory: Path) -> None:
    """Remove only manifests owned by this command before rebuilding them."""

    for filename in (
        "waymo_evaluation_setup.json",
        "reconstruction_manifest.json",
        "novel_view_manifest.json",
    ):
        (output_directory / filename).unlink(missing_ok=True)


def _metric_manifest(
    frames: Sequence[Mapping[str, Any]],
    *,
    output_directory: Path,
    evaluation_name: str,
) -> dict[str, Any]:
    pairs = []
    for frame in frames:
        target = Path(str(frame["target"]))
        prediction = (
            output_directory
            / "renders"
            / evaluation_name
            / str(frame["camera"])
            / f"{int(frame['source_frame_index']):06d}.png"
        )
        pairs.append(
            {
                "prediction": str(prediction.relative_to(output_directory)),
                "target": str(target.relative_to(output_directory)),
                "frame_index": int(frame["frame_index"]),
                "source_frame_index": int(frame["source_frame_index"]),
                "timestamp_micros": int(frame["timestamp_micros"]),
                "camera": str(frame["camera"]),
            }
        )
    return {
        "dataset": "waymo_v2",
        "evaluation": evaluation_name,
        "metric_protocols": dict(_METRIC_PROTOCOLS),
        "pairs": pairs,
    }


def prepare_waymo_v2_evaluation(
    *,
    waymo_root: Path,
    parquet_dir: str,
    sequence: str,
    cameras: Sequence[str],
    start_frame: int,
    end_frame: int | None,
    test_every: int,
    first_test_position: int,
    target_size: tuple[int, int],
    output_directory: Path,
    extract_images: bool,
) -> dict[str, Any]:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if test_every <= 0:
        raise ValueError("test_every must be positive")
    if first_test_position < 0:
        raise ValueError("first_test_position must be non-negative")
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("target_size must be positive")
    normalized_cameras = tuple(str(camera).upper() for camera in cameras)
    unknown = set(normalized_cameras) - set(WAYMO_CAMERA_IDS)
    if unknown or not normalized_cameras:
        raise ValueError(f"invalid Waymo camera selection: {sorted(unknown)}")
    if len(normalized_cameras) != len(set(normalized_cameras)):
        raise ValueError("camera selection must be unique")

    root = waymo_root.resolve(strict=True)
    paths = _component_paths(
        root,
        parquet_dir=parquet_dir,
        sequence=sequence,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    _invalidate_generated_manifests(output_directory)
    _validate_component_schemas(paths)
    requested_ids = {WAYMO_CAMERA_IDS[camera] for camera in normalized_cameras}
    calibration = _calibrations(
        paths["camera_calibration"],
        requested_ids=requested_ids,
        target_size=target_size,
    )
    pose_rows = _read_parquet_rows(
        paths["vehicle_pose"],
        (_TIMESTAMP_COLUMN,),
    )
    pose_timestamps = [int(row[_TIMESTAMP_COLUMN]) for row in pose_rows]
    if len(pose_timestamps) != len(set(pose_timestamps)):
        raise ValueError("vehicle_pose contains duplicate capture timestamps")
    timestamps = sorted(pose_timestamps)
    if not timestamps:
        raise ValueError("vehicle_pose contains no capture timestamps")
    if start_frame >= len(timestamps):
        raise ValueError(
            f"start_frame {start_frame} is outside {len(timestamps)} captures"
        )
    if end_frame is not None and end_frame >= len(timestamps):
        raise ValueError(
            f"end_frame {end_frame} is outside {len(timestamps)} captures"
        )

    image_rows = _read_parquet_rows(
        paths["camera_image"],
        (
            (_TIMESTAMP_COLUMN, _CAMERA_COLUMN, _IMAGE_COLUMN)
            if extract_images
            else (_TIMESTAMP_COLUMN, _CAMERA_COLUMN)
        ),
        filters=[(_CAMERA_COLUMN, "in", sorted(requested_ids))],
    )
    selected_timestamps = timestamps[
        start_frame : (end_frame + 1 if end_frame is not None else None)
    ]
    if not selected_timestamps:
        raise ValueError("selected Waymo frame range is empty")
    timestamp_to_source_index = {
        timestamp: index for index, timestamp in enumerate(timestamps)
    }
    timestamp_to_frame_index = {
        timestamp: index for index, timestamp in enumerate(selected_timestamps)
    }
    channel_by_id = {
        camera_id: channel for channel, camera_id in WAYMO_CAMERA_IDS.items()
    }
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    selected_set = set(selected_timestamps)
    for row in image_rows:
        timestamp = int(row[_TIMESTAMP_COLUMN])
        camera_id = int(row[_CAMERA_COLUMN])
        if timestamp not in selected_set or camera_id not in requested_ids:
            continue
        key = (timestamp, camera_id)
        if key in rows_by_key:
            raise ValueError(
                f"duplicate camera image for timestamp/camera {key}"
            )
        rows_by_key[key] = row
    expected_keys = {
        (timestamp, camera_id)
        for timestamp in selected_timestamps
        for camera_id in requested_ids
    }
    missing_keys = expected_keys - set(rows_by_key)
    if missing_keys:
        raise ValueError(
            "selected capture is missing requested camera rows: "
            + ", ".join(str(key) for key in sorted(missing_keys))
        )

    frames: list[dict[str, Any]] = []
    for timestamp in selected_timestamps:
        relative_index = timestamp_to_frame_index[timestamp]
        source_index = timestamp_to_source_index[timestamp]
        split = (
            "novel_view"
            if relative_index >= first_test_position
            and (relative_index - first_test_position) % test_every == 0
            else "reconstruction"
        )
        for camera in normalized_cameras:
            raw_camera_id = WAYMO_CAMERA_IDS[camera]
            target_path = (
                output_directory
                / "targets"
                / camera
                / f"{source_index:06d}.png"
            )
            if extract_images:
                encoded = rows_by_key[(timestamp, raw_camera_id)][_IMAGE_COLUMN]
                if not isinstance(encoded, (bytes, bytearray)):
                    raise ValueError("Waymo camera image column must contain bytes")
                source_height, source_width = calibration[raw_camera_id][
                    "source_size"
                ]
                resized_height, resized_width = calibration[raw_camera_id][
                    "target_size"
                ]
                _atomic_save_png(
                    bytes(encoded),
                    target_path,
                    source_size=(source_height, source_width),
                    target_size=(resized_height, resized_width),
                )
            frames.append(
                {
                    "frame_index": relative_index,
                    "source_frame_index": source_index,
                    "timestamp_micros": timestamp,
                    "camera": camera,
                    "camera_id": raw_camera_id - 1,
                    "waymo_camera_id": raw_camera_id,
                    "split": split,
                    "target": str(target_path) if extract_images else None,
                }
            )

    reconstruction = [
        frame for frame in frames if frame["split"] == "reconstruction"
    ]
    novel_view = [frame for frame in frames if frame["split"] == "novel_view"]
    if not novel_view:
        raise ValueError(
            "Waymo split selects no novel-view frames; expand the frame range "
            "or lower first_test_position"
        )
    if not reconstruction:
        raise ValueError(
            "Waymo split selects no reconstruction frames; increase "
            "first_test_position or test_every"
        )
    setup = {
        "dataset": "waymo_v2",
        "sequence": sequence,
        "waymo_root": str(root),
        "parquet_dir": parquet_dir,
        "component_paths": {
            component: str(path) for component, path in paths.items()
        },
        "camera_channels": list(normalized_cameras),
        "calibration": {
            channel_by_id[camera_id]: value
            for camera_id, value in sorted(calibration.items())
        },
        "selected_source_range": {
            "start": start_frame,
            "end": (
                end_frame if end_frame is not None else len(timestamps) - 1
            ),
            "end_inclusive": True,
        },
        "split_protocol": {
            "ordering": "capture timestamp ascending",
            "test_every": test_every,
            "first_test_position": first_test_position,
            "capture_atomic": True,
            "paper_basis": (
                "ArmGS Sec. 4.1 follows StreetGaussians; official StreetGS "
                "get_val_frames selects range(test_every, N, test_every)"
            ),
        },
        "evaluation": {
            "reconstruction_split": "training views",
            "novel_view_split": "held-out testing views",
            "metric_protocols": dict(_METRIC_PROTOCOLS),
            "target_generation": {
                "decode": "Pillow RGB",
                "resize": "uniform width-derived scale; no upsampling",
                "resize_kernel": "Pillow BILINEAR",
                "encoding": "lossless PNG",
                "intrinsics_scale": "same uniform factor for both K rows",
            },
        },
        "counts": {
            "captures": len(selected_timestamps),
            "camera_images": len(frames),
            "reconstruction_captures": len(
                {frame["frame_index"] for frame in reconstruction}
            ),
            "reconstruction_images": len(reconstruction),
            "novel_view_captures": len(
                {frame["frame_index"] for frame in novel_view}
            ),
            "novel_view_images": len(novel_view),
        },
        "frames": frames,
        "training_adapter_status": {
            "camera_evaluation_assets": "ready" if extract_images else "indexed",
            "lidar_canonicalization": "not_performed_by_this_script",
            "actor_track_canonicalization": "not_performed_by_this_script",
            "sfm_initialization": "not_performed_by_this_script",
        },
    }
    _write_json(output_directory / "waymo_evaluation_setup.json", setup)
    if extract_images:
        _write_json(
            output_directory / "reconstruction_manifest.json",
            _metric_manifest(
                reconstruction,
                output_directory=output_directory,
                evaluation_name="reconstruction",
            ),
        )
        _write_json(
            output_directory / "novel_view_manifest.json",
            _metric_manifest(
                novel_view,
                output_directory=output_directory,
                evaluation_name="novel_view",
            ),
        )
    return setup


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        setup = prepare_waymo_v2_evaluation(
            waymo_root=args.waymo_root,
            parquet_dir=args.parquet_dir,
            sequence=args.sequence,
            cameras=args.cameras,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            test_every=args.test_every,
            first_test_position=args.first_test_position,
            target_size=(args.target_height, args.target_width),
            output_directory=args.output_dir,
            extract_images=args.extract_images,
        )
        print(
            json.dumps(
                {
                    "output": str(
                        args.output_dir / "waymo_evaluation_setup.json"
                    ),
                    **setup["counts"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
