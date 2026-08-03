"""Canonical data contracts and dataset-specific import adapters."""

from .kitti import (
    KittiCalibration,
    KittiTracklet,
    KittiTrackletPose,
    canonicalize_kitti_tracklets,
    load_kitti_manifest,
    parse_kitti_calibration,
    parse_kitti_poses,
    parse_kitti_timestamps,
    parse_kitti_tracklets,
    project_velodyne_to_image,
    read_png_size,
    read_velodyne_bin,
)
from .schema import (
    ActorTrack,
    ActorTrackSample,
    CameraConvention,
    CanonicalDatasetManifest,
    CanonicalFrame,
    CanonicalFrameDataset,
    LidarFrame,
    LidarProjection,
)

__all__ = [
    "ActorTrack",
    "ActorTrackSample",
    "CameraConvention",
    "CanonicalDatasetManifest",
    "CanonicalFrame",
    "CanonicalFrameDataset",
    "KittiCalibration",
    "KittiTracklet",
    "KittiTrackletPose",
    "LidarFrame",
    "LidarProjection",
    "canonicalize_kitti_tracklets",
    "load_kitti_manifest",
    "parse_kitti_calibration",
    "parse_kitti_poses",
    "parse_kitti_timestamps",
    "parse_kitti_tracklets",
    "project_velodyne_to_image",
    "read_png_size",
    "read_velodyne_bin",
]
