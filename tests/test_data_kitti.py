from __future__ import annotations

from pathlib import Path
import struct

import pytest
import torch

from armgs.data import (
    CanonicalFrame,
    CanonicalFrameDataset,
    KittiTracklet,
    KittiTrackletPose,
    canonicalize_kitti_tracklets,
    load_kitti_manifest,
    parse_kitti_calibration,
    parse_kitti_poses,
    parse_kitti_timestamps,
    parse_kitti_tracklets,
    project_velodyne_to_image,
    read_velodyne_bin,
)


def _projection(last_x: float = 0.0) -> str:
    values = [10.0, 0.0, 2.0, last_x, 0.0, 10.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return " ".join(str(value) for value in values)


def _write_calibration(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                f"P0: {_projection()}",
                f"P1: {_projection(-1.0)}",
                f"P2: {_projection(-2.0)}",
                f"P3: {_projection(-3.0)}",
                "R0_rect: 1 0 0 0 1 0 0 0 1",
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0",
            )
        ),
        encoding="utf-8",
    )


def _write_png_header(path: Path, height: int = 4, width: int = 6) -> None:
    # The loader only needs the standard signature and IHDR dimensions.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


def _write_scan(path: Path) -> None:
    rows = (
        (0.0, 0.0, 2.0, 0.5),  # inside camera 2: (u,v)=(1,2)
        (2.0, 0.0, 1.0, 0.6),  # outside image
        (0.0, 0.0, -1.0, 0.7),  # behind camera
    )
    path.write_bytes(b"".join(struct.pack("<ffff", *row) for row in rows))


def _write_tracklets(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<boost_serialization>
  <tracklets>
    <count>1</count>
    <item_version>1</item_version>
    <item>
      <objectType>Car</objectType>
      <h>1.5</h><w>1.8</w><l>4.2</l>
      <first_frame>0</first_frame>
      <poses>
        <count>2</count><item_version>2</item_version>
        <item>
          <tx>1</tx><ty>0</ty><tz>0</tz>
          <rx>0</rx><ry>0</ry><rz>0</rz>
          <occlusion>0</occlusion><truncation>0</truncation>
        </item>
        <item>
          <tx>2</tx><ty>0</ty><tz>0</tz>
          <rx>0</rx><ry>0</ry><rz>1.5707963267948966</rz>
          <occlusion>1</occlusion><truncation>0</truncation>
        </item>
      </poses>
    </item>
  </tracklets>
</boost_serialization>
""",
        encoding="utf-8",
    )


def _make_sequence(root: Path, *, masks: bool = False) -> dict[str, Path]:
    root.mkdir()
    _write_calibration(root / "calib.txt")
    (root / "poses.txt").write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n"
        "1 0 0 10 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )
    (root / "times.txt").write_text("0.0\n0.1\n", encoding="utf-8")
    image_dir = root / "image_2"
    lidar_dir = root / "velodyne"
    image_dir.mkdir()
    lidar_dir.mkdir()
    for frame_index in range(2):
        _write_png_header(image_dir / f"{frame_index:06d}.png")
        _write_scan(lidar_dir / f"{frame_index:06d}.bin")
    tracklets = root / "tracklet_labels.xml"
    _write_tracklets(tracklets)
    result = {"tracklets": tracklets}
    if masks:
        for label in ("sky", "actor"):
            directory = root / f"{label}_masks"
            directory.mkdir()
            for frame_index in range(2):
                _write_png_header(directory / f"{frame_index:06d}.png")
            result[label] = directory
    return result


def test_timestamp_parser_preserves_nanosecond_precision(tmp_path: Path) -> None:
    path = tmp_path / "times.txt"
    path.write_text(
        "1700000000.000000001\n1700000000.000000002\n", encoding="utf-8"
    )

    timestamps = parse_kitti_timestamps(path)

    assert timestamps.dtype == torch.int64
    assert timestamps.tolist() == [1700000000000000001, 1700000000000000002]


def test_calibration_pose_and_camera_baseline_are_explicit(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calib.txt"
    poses_path = tmp_path / "poses.txt"
    _write_calibration(calibration_path)
    poses_path.write_text("1 0 0 10 0 1 0 0 0 0 1 0\n", encoding="utf-8")

    calibration = parse_kitti_calibration(calibration_path)
    pose = parse_kitti_poses(poses_path)[0]
    camera2_to_world = calibration.camera_to_world(pose, 2)

    assert torch.allclose(calibration.intrinsics(2), torch.tensor(
        [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ))
    # P2 has K*t=(-2,0,0), so camera 2's center is +0.2 m from camera 0.
    assert torch.allclose(camera2_to_world[:3, 3], torch.tensor([10.2, 0.0, 0.0], dtype=torch.float64))


def test_velodyne_reader_and_projection_filter_invalid_samples(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calib.txt"
    scan_path = tmp_path / "000000.bin"
    _write_calibration(calibration_path)
    _write_scan(scan_path)

    points, reflectance = read_velodyne_bin(scan_path)
    projected = project_velodyne_to_image(
        points, parse_kitti_calibration(calibration_path), 2, (4, 6)
    )

    assert points.shape == (3, 3)
    assert torch.allclose(reflectance, torch.tensor([0.5, 0.6, 0.7]))
    assert projected.source_point_indices.tolist() == [0]
    assert torch.allclose(projected.image_coordinates, torch.tensor([[1.0, 2.0]]))
    assert projected.pixel_indices.tolist() == [[1, 2]]
    assert projected.depths.tolist() == [2.0]


def test_tracklets_become_world_tracks_with_lifecycle(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calib.txt"
    tracklet_path = tmp_path / "tracklet_labels.xml"
    _write_calibration(calibration_path)
    _write_tracklets(tracklet_path)
    poses = (
        torch.eye(4, dtype=torch.float64),
        torch.tensor(
            [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float64,
        ),
    )
    timestamps = torch.tensor([0, 100_000_000], dtype=torch.int64)

    raw = parse_kitti_tracklets(tracklet_path)
    tracks = canonicalize_kitti_tracklets(
        raw, timestamps, poses, parse_kitti_calibration(calibration_path)
    )

    assert len(tracks) == 1
    track = tracks[0]
    assert track.class_name == "Car"
    assert torch.allclose(track.dimensions_lwh, torch.tensor([4.2, 1.8, 1.5], dtype=torch.float64))
    # KITTI translation is a bottom center; canonical translation is box center.
    assert torch.allclose(track.samples[0].translation, torch.tensor([1.0, 0.0, 0.75], dtype=torch.float64))
    assert torch.allclose(track.samples[1].translation, torch.tensor([12.0, 0.0, 0.75], dtype=torch.float64))
    assert tuple(int(value.item()) for value in track.lifecycle_timestamps) == (0, 100_000_000)
    assert torch.allclose(
        track.samples[1].quaternion_wxyz.abs(),
        torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=torch.float64),
        atol=1.0e-7,
    )


def test_tracklet_bottom_center_offset_follows_rotated_lidar_height_axis(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calib.txt"
    _write_calibration(calibration_path)
    tracklet = KittiTracklet(
        object_type="Car",
        dimensions_hwl=torch.tensor([2.0, 1.5, 4.0], dtype=torch.float64),
        first_frame=0,
        poses=(
            KittiTrackletPose(
                frame_index=0,
                translation_lidar=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
                rotation_rpy=torch.tensor(
                    [torch.pi / 2.0, 0.0, 0.0], dtype=torch.float64
                ),
            ),
        ),
    )

    track = canonicalize_kitti_tracklets(
        (tracklet,),
        torch.tensor([0], dtype=torch.int64),
        (torch.eye(4, dtype=torch.float64),),
        parse_kitti_calibration(calibration_path),
    )[0]

    assert torch.allclose(
        track.dimensions_lwh,
        torch.tensor([4.0, 1.5, 2.0], dtype=torch.float64),
    )
    # R_x(+pi/2) maps the local +z height offset to Velodyne -y.
    assert torch.allclose(
        track.samples[0].translation,
        torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64),
        atol=1.0e-7,
    )


def test_load_manifest_is_indexable_and_aligns_optional_masks(tmp_path: Path) -> None:
    paths = _make_sequence(tmp_path / "sequence", masks=True)

    manifest = load_kitti_manifest(
        tmp_path / "sequence",
        camera_ids=(2,),
        tracklet_path=paths["tracklets"],
        sky_mask_dirs={2: paths["sky"]},
        actor_mask_dirs={2: paths["actor"]},
    )
    dataset = CanonicalFrameDataset(manifest)

    assert len(dataset) == 2
    assert manifest.timestamp_unit == "nanoseconds"
    frame = dataset[1]
    assert isinstance(frame, CanonicalFrame)
    assert frame.timestamp.dtype == torch.int64
    assert frame.timestamp.item() == 100_000_000
    assert frame.camera_id == 2
    assert frame.camera_convention == "opencv"
    assert frame.image_size == (4, 6)
    assert frame.image_path.name == "000001.png"
    assert frame.sky_mask_path is not None and frame.actor_mask_path is not None
    assert frame.lidar is not None and frame.lidar_projection is not None
    assert frame.lidar.points.shape == (1, 3)
    assert torch.allclose(frame.lidar.world_points[0], torch.tensor([10.0, 0.0, 2.0]))
    assert len(manifest.actor_tracks) == 1



def test_manifest_can_retain_complete_raw_lidar_scan(tmp_path: Path) -> None:
    _make_sequence(tmp_path / "sequence")

    manifest = load_kitti_manifest(
        tmp_path / "sequence",
        camera_ids=(2,),
        retain_unprojected_lidar=True,
    )

    for frame in manifest:
        assert frame.lidar is not None
        assert frame.lidar.points.shape == (3, 3)

def test_missing_and_malformed_inputs_fail_early(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="calibration"):
        parse_kitti_calibration(tmp_path / "missing.txt")

    calibration_path = tmp_path / "bad_calib.txt"
    _write_calibration(calibration_path)
    calibration_path.write_text(
        calibration_path.read_text(encoding="utf-8").replace("P3:", "PX:"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing calibration key P3"):
        parse_kitti_calibration(calibration_path)

    scan_path = tmp_path / "bad.bin"
    scan_path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="multiple of 16"):
        read_velodyne_bin(scan_path)


def test_manifest_requires_every_requested_lidar_scan(tmp_path: Path) -> None:
    _make_sequence(tmp_path / "sequence")
    (tmp_path / "sequence" / "velodyne" / "000001.bin").unlink()

    with pytest.raises(FileNotFoundError, match="000001.bin"):
        load_kitti_manifest(tmp_path / "sequence", camera_ids=(2,))

