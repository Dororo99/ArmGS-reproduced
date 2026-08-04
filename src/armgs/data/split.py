"""Leak-free train/evaluation splits for canonical multi-camera datasets.

Rows that share a canonical ``frame_index`` are treated as one capture and are
always assigned together.  The module also prepares the exact row metadata
consumed by ``FrameAppearanceEmbedding`` and ``NearestFrameLookup``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from .schema import ActorTrack, CanonicalDatasetManifest


@dataclass(frozen=True)
class CanonicalDatasetSplit:
    """A validated split plus training-embedding lookup metadata.

    ``training_camera_ids`` and ``training_timestamps`` have one row per frame
    in ``train_manifest``.  Consequently, their row numbers directly index a
    ``FrameAppearanceEmbedding``. ``source_index_to_training_row`` maps row
    indices from the unsplit source manifest to those embedding rows.

    Actor pose knots are sliced independently for each manifest, while their
    unsplit raw lifecycle bounds are retained. A training trajectory must be
    built from ``train_manifest``; ``eval_manifest`` tracks supply exact raw
    boxes at held-out captures for evaluation only.
    """

    train_manifest: CanonicalDatasetManifest
    eval_manifest: CanonicalDatasetManifest
    training_camera_ids: Tensor
    training_timestamps: Tensor
    train_source_indices: tuple[int, ...]
    eval_source_indices: tuple[int, ...]
    source_index_to_training_row: Mapping[int, int]

    def __post_init__(self) -> None:
        train_count = len(self.train_manifest)
        if self.training_camera_ids.shape != (train_count,):
            raise ValueError("training_camera_ids must have one row per training frame")
        if self.training_timestamps.shape != (train_count,):
            raise ValueError("training_timestamps must have one row per training frame")
        if self.training_camera_ids.dtype != torch.long:
            raise ValueError("training_camera_ids must have dtype torch.long")
        if self.training_timestamps.dtype != torch.int64:
            raise ValueError("training_timestamps must have dtype torch.int64")
        if (
            not self.training_camera_ids.is_contiguous()
            or not self.training_timestamps.is_contiguous()
        ):
            raise ValueError("training lookup tensors must be contiguous")
        if len(self.train_source_indices) != train_count:
            raise ValueError("train_source_indices must align with train_manifest")
        if len(self.eval_source_indices) != len(self.eval_manifest):
            raise ValueError("eval_source_indices must align with eval_manifest")
        expected_mapping = {
            source_index: training_row
            for training_row, source_index in enumerate(self.train_source_indices)
        }
        if dict(self.source_index_to_training_row) != expected_mapping:
            raise ValueError("source_index_to_training_row is inconsistent")
        object.__setattr__(
            self,
            "source_index_to_training_row",
            MappingProxyType(expected_mapping),
        )

    def training_row(self, source_index: int) -> int:
        """Return an appearance-embedding row for a source-manifest row."""

        try:
            return self.source_index_to_training_row[source_index]
        except KeyError as error:
            raise ValueError(
                f"source manifest row {source_index} is not in the training split"
            ) from error


def _capture_groups(
    manifest: CanonicalDatasetManifest,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Return ``(frame_index, capture timestamp, source rows)`` in order.

    Camera observation timestamps are intentionally allowed to differ within
    a capture. The nominal capture timestamp, not sensor time, establishes the
    atomic split identity.
    """

    capture_timestamp_by_frame_index: dict[int, int] = {}
    frame_index_by_capture_timestamp: dict[int, int] = {}
    rows_by_frame_index: dict[int, list[int]] = {}
    seen_capture_camera: set[tuple[int, int]] = set()

    for source_index, frame in enumerate(manifest.frames):
        capture_timestamp_tensor = frame.capture_timestamp
        if capture_timestamp_tensor is None:  # CanonicalFrame normalizes this.
            raise RuntimeError("canonical frame is missing capture_timestamp")
        capture_timestamp = int(capture_timestamp_tensor.item())
        key = (frame.frame_index, frame.camera_id)
        if key in seen_capture_camera:
            raise ValueError(
                "duplicate canonical frame for frame_index/camera_id "
                f"({frame.frame_index}, {frame.camera_id})"
            )
        seen_capture_camera.add(key)

        previous_timestamp = capture_timestamp_by_frame_index.setdefault(
            frame.frame_index, capture_timestamp
        )
        if previous_timestamp != capture_timestamp:
            raise ValueError(
                f"frame_index {frame.frame_index} has multiple capture timestamps"
            )
        previous_frame_index = frame_index_by_capture_timestamp.setdefault(
            capture_timestamp, frame.frame_index
        )
        if previous_frame_index != frame.frame_index:
            raise ValueError(
                f"capture timestamp {capture_timestamp} has multiple frame_index values"
            )
        rows_by_frame_index.setdefault(frame.frame_index, []).append(source_index)

    return tuple(
        (
            frame_index,
            capture_timestamp_by_frame_index[frame_index],
            tuple(rows_by_frame_index[frame_index]),
        )
        for frame_index in sorted(
            rows_by_frame_index,
            key=lambda index: (capture_timestamp_by_frame_index[index], index),
        )
    )


def _make_manifest(
    source: CanonicalDatasetManifest, source_indices: tuple[int, ...]
) -> CanonicalDatasetManifest:
    selected_frame_indices = {
        source.frames[index].frame_index for index in source_indices
    }
    actor_tracks = tuple(
        ActorTrack(
            actor_id=track.actor_id,
            class_name=track.class_name,
            dimensions_lwh=track.dimensions_lwh,
            samples=selected_samples,
            lifecycle_start_timestamp=track.lifecycle_start_timestamp,
            lifecycle_end_timestamp=track.lifecycle_end_timestamp,
        )
        for track in source.actor_tracks
        if (
            selected_samples := tuple(
                sample
                for sample in track.samples
                if sample.frame_index in selected_frame_indices
            )
        )
    )
    return CanonicalDatasetManifest(
        frames=tuple(source.frames[index] for index in source_indices),
        actor_tracks=actor_tracks,
        timestamp_unit=source.timestamp_unit,
    )


def _build_split(
    manifest: CanonicalDatasetManifest,
    groups: tuple[tuple[int, int, tuple[int, ...]], ...],
    eval_frame_indices: frozenset[int],
) -> CanonicalDatasetSplit:
    known_frame_indices = {frame_index for frame_index, _, _ in groups}
    unknown = eval_frame_indices - known_frame_indices
    if unknown:
        raise ValueError(f"unknown evaluation frame_index values: {sorted(unknown)}")
    if not eval_frame_indices:
        raise ValueError("evaluation split must contain at least one capture")
    if eval_frame_indices == known_frame_indices:
        raise ValueError("training split must contain at least one capture")

    eval_rows = {
        source_index
        for frame_index, _, source_indices in groups
        if frame_index in eval_frame_indices
        for source_index in source_indices
    }
    # Retain source-manifest row order, independent of how captures were sorted.
    train_source_indices = tuple(
        index for index in range(len(manifest)) if index not in eval_rows
    )
    eval_source_indices = tuple(
        index for index in range(len(manifest)) if index in eval_rows
    )
    train_manifest = _make_manifest(manifest, train_source_indices)
    eval_manifest = _make_manifest(manifest, eval_source_indices)

    train_camera_set = {frame.camera_id for frame in train_manifest.frames}
    eval_camera_set = {frame.camera_id for frame in eval_manifest.frames}
    missing_cameras = eval_camera_set - train_camera_set
    if missing_cameras:
        raise ValueError(
            "evaluation camera ids have no training frames for nearest lookup: "
            f"{sorted(missing_cameras)}"
        )

    training_camera_ids = torch.tensor(
        [frame.camera_id for frame in train_manifest.frames], dtype=torch.long
    ).contiguous()
    training_timestamps = torch.stack(
        [
            frame.timestamp.detach().to(device="cpu", dtype=torch.int64)
            for frame in train_manifest.frames
        ]
    ).contiguous()
    source_to_row = {
        source_index: row
        for row, source_index in enumerate(train_source_indices)
    }
    return CanonicalDatasetSplit(
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        training_camera_ids=training_camera_ids,
        training_timestamps=training_timestamps,
        train_source_indices=train_source_indices,
        eval_source_indices=eval_source_indices,
        source_index_to_training_row=source_to_row,
    )


def split_manifest_by_frame_indices(
    manifest: CanonicalDatasetManifest,
    eval_frame_indices: Iterable[int],
) -> CanonicalDatasetSplit:
    """Hold out complete captures selected by canonical ``frame_index``.

    Explicit indices must be unique integers present in ``manifest``. Both
    sides of the resulting split are required to be non-empty.
    """

    requested = tuple(eval_frame_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, int) for index in requested
    ):
        raise TypeError("eval_frame_indices must contain integers")
    if len(requested) != len(set(requested)):
        raise ValueError("eval_frame_indices must be unique")
    groups = _capture_groups(manifest)
    return _build_split(manifest, groups, frozenset(requested))


def periodic_train_eval_split(
    manifest: CanonicalDatasetManifest,
    *,
    every: int = 8,
    offset: int = 0,
    start_position: int = 0,
) -> CanonicalDatasetSplit:
    """Hold out every ``every``-th capture in timestamp order.

    ``offset`` is an ordinal in ``[0, every)``. Selection is based on capture
    position, not the numeric value of a possibly sparse ``frame_index``.
    ``start_position`` keeps earlier captures in the training split before the
    periodic schedule begins. This represents protocols such as StreetGS Waymo,
    whose held-out positions are ``4, 8, 12, ...`` rather than
    ``0, 4, 8, ...``.
    """

    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError("every must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be an integer")
    if offset < 0 or offset >= every:
        raise ValueError("offset must satisfy 0 <= offset < every")
    if (
        isinstance(start_position, bool)
        or not isinstance(start_position, int)
        or start_position < 0
    ):
        raise ValueError("start_position must be a non-negative integer")
    groups = _capture_groups(manifest)
    selected = frozenset(
        frame_index
        for position, (frame_index, _, _) in enumerate(groups)
        if position >= start_position and position % every == offset
    )
    return _build_split(manifest, groups, selected)


__all__ = [
    "CanonicalDatasetSplit",
    "periodic_train_eval_split",
    "split_manifest_by_frame_indices",
]
