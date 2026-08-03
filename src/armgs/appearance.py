"""Local and global appearance refinement from ArmGS equations (3)--(6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encodings import HashGridEncoder, SinusoidalEncoder
from .networks import MLP


@dataclass(frozen=True)
class LocalAffine:
    scale: Tensor
    bias: Tensor


@dataclass(frozen=True)
class GlobalAffine:
    matrix: Tensor
    bias: Tensor


class FrameAppearanceEmbedding(nn.Module):
    """One learnable low-dimensional appearance code per training image."""

    def __init__(self, num_training_frames: int, embedding_dim: int = 16) -> None:
        super().__init__()
        if num_training_frames <= 0 or embedding_dim <= 0:
            raise ValueError("embedding table dimensions must be positive")
        self.embedding = nn.Embedding(num_training_frames, embedding_dim)
        # A tiny perturbation lets frames separate while remaining close to the
        # identity-initialized 3DGS baseline.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0e-4)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, frame_indices: Tensor) -> Tensor:
        return self.embedding(
            frame_indices.to(device=self.embedding.weight.device, dtype=torch.long)
        )


class NearestFrameLookup(nn.Module):
    """Paper-specified novel-view lookup by camera id and timestamp.

    Returned indices address rows in :class:`FrameAppearanceEmbedding`, not
    external dataset frame identifiers.
    """

    def __init__(self, camera_ids: Tensor, timestamps: Tensor) -> None:
        super().__init__()
        if camera_ids.ndim != 1 or timestamps.ndim != 1:
            raise ValueError("camera_ids and timestamps must be one-dimensional")
        if camera_ids.shape != timestamps.shape or camera_ids.numel() == 0:
            raise ValueError("camera_ids and timestamps must be non-empty and aligned")
        self.register_buffer("camera_ids", camera_ids.to(dtype=torch.long))
        self.register_buffer("timestamps", timestamps.to(dtype=torch.float64))

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        # Preserve absolute-time precision when the surrounding model is cast.
        timestamps = self.timestamps
        result = super()._apply(fn)
        self._buffers["timestamps"] = timestamps.to(
            device=self.timestamps.device, dtype=torch.float64
        )
        return result

    def forward(self, query_camera_ids: Tensor, query_timestamps: Tensor) -> Tensor:
        scalar_query = query_camera_ids.ndim == 0 and query_timestamps.ndim == 0
        query_camera_ids = query_camera_ids.reshape(-1).to(
            device=self.camera_ids.device, dtype=torch.long
        )
        query_timestamps = query_timestamps.reshape(-1).to(
            device=self.timestamps.device, dtype=self.timestamps.dtype
        )
        if query_camera_ids.shape != query_timestamps.shape:
            raise ValueError("query camera ids and timestamps must be aligned")

        same_camera = query_camera_ids[:, None] == self.camera_ids[None, :]
        missing = ~same_camera.any(dim=1)
        if missing.any():
            missing_ids = query_camera_ids[missing].detach().cpu().tolist()
            raise ValueError(
                "no training frame exists for query camera id(s) " f"{missing_ids}"
            )
        distances = torch.abs(query_timestamps[:, None] - self.timestamps[None, :])
        distances = distances.masked_fill(~same_camera, torch.inf)
        result = distances.argmin(dim=1)
        return result[0] if scalar_query else result


class LocalGaussianAppearanceRefiner(nn.Module):
    """Gaussian-wise RGB affine transform from equations (3) and (4)."""

    _COLOR_DIM: Final[int] = 3

    def __init__(
        self,
        position_encoder: HashGridEncoder,
        frame_embedding_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 3,
        hash_chunk_size: int | None = None,
        scale_delta_limit: float | None = None,
        bias_limit: float | None = None,
    ) -> None:
        super().__init__()
        if frame_embedding_dim <= 0:
            raise ValueError("frame_embedding_dim must be positive")
        if hash_chunk_size is not None and hash_chunk_size <= 0:
            raise ValueError("hash_chunk_size must be positive")
        if scale_delta_limit is not None and scale_delta_limit <= 0:
            raise ValueError("scale_delta_limit must be positive")
        if bias_limit is not None and bias_limit <= 0:
            raise ValueError("bias_limit must be positive")
        self.position_encoder = position_encoder
        self.hash_chunk_size = hash_chunk_size
        self.scale_delta_limit = scale_delta_limit
        self.bias_limit = bias_limit
        input_dim = position_encoder.output_dim + frame_embedding_dim + self._COLOR_DIM
        self.affine_learner = MLP(input_dim, hidden_dim, 6, num_layers)
        self._initialize_identity()

    def _initialize_identity(self) -> None:
        final = self.affine_learner.final_layer
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.zero_()
            final.bias[:3].fill_(1.0)

    @staticmethod
    def _broadcast_frame_embedding(frame_embedding: Tensor, count: int) -> Tensor:
        if frame_embedding.ndim == 1:
            return frame_embedding.unsqueeze(0).expand(count, -1)
        if frame_embedding.ndim != 2:
            raise ValueError("frame_embedding must have shape [E], [1,E], or [N,E]")
        if frame_embedding.shape[0] == 1:
            return frame_embedding.expand(count, -1)
        if frame_embedding.shape[0] != count:
            raise ValueError("per-Gaussian frame embeddings must have leading size N")
        return frame_embedding

    def predict_affine(
        self, positions: Tensor, colors: Tensor, frame_embedding: Tensor
    ) -> LocalAffine:
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N,3]")
        if colors.shape != positions.shape:
            raise ValueError("colors must have shape [N,3]")
        expanded_embedding = self._broadcast_frame_embedding(
            frame_embedding, positions.shape[0]
        ).to(colors)
        position_features = self.position_encoder(
            positions, chunk_size=self.hash_chunk_size
        ).to(colors)
        features = torch.cat((position_features, expanded_embedding, colors), dim=-1)
        affine = self.affine_learner(features)
        scale = affine[..., :3]
        bias = affine[..., 3:]
        if self.scale_delta_limit is not None:
            limit = self.scale_delta_limit
            scale = 1.0 + limit * torch.tanh((scale - 1.0) / limit)
        if self.bias_limit is not None:
            limit = self.bias_limit
            bias = limit * torch.tanh(bias / limit)
        return LocalAffine(scale=scale, bias=bias)

    def forward(
        self, positions: Tensor, colors: Tensor, frame_embedding: Tensor
    ) -> Tensor:
        affine = self.predict_affine(positions, colors, frame_embedding)
        return affine.scale * colors + affine.bias


class ViewpointEncoder(nn.Module):
    """Configurable code for camera center and optical viewing direction."""

    def __init__(
        self,
        *,
        position_frequencies: int = 6,
        direction_frequencies: int = 4,
        aabb_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        aabb_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        minimum = torch.tensor(aabb_min, dtype=torch.float32)
        maximum = torch.tensor(aabb_max, dtype=torch.float32)
        if torch.any(maximum <= minimum):
            raise ValueError("aabb_max must be greater than aabb_min")
        self.register_buffer("aabb_min", minimum)
        self.register_buffer("aabb_max", maximum)
        self.position_encoder = SinusoidalEncoder(3, position_frequencies)
        self.direction_encoder = SinusoidalEncoder(3, direction_frequencies)

    @property
    def output_dim(self) -> int:
        return self.position_encoder.output_dim + self.direction_encoder.output_dim

    def set_aabb(self, minimum: Tensor, maximum: Tensor) -> None:
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("AABB tensors must have shape [3]")
        if torch.any(maximum <= minimum):
            raise ValueError("maximum must be greater than minimum")
        self.aabb_min.copy_(minimum.to(self.aabb_min))
        self.aabb_max.copy_(maximum.to(self.aabb_max))

    def forward(self, camera_centers: Tensor, view_directions: Tensor) -> Tensor:
        if camera_centers.shape[-1] != 3 or view_directions.shape[-1] != 3:
            raise ValueError("camera centers and view directions must end in dimension 3")
        if camera_centers.shape != view_directions.shape:
            raise ValueError("camera centers and view directions must have matching shapes")
        minimum = self.aabb_min.to(camera_centers)
        maximum = self.aabb_max.to(camera_centers)
        normalized_position = 2.0 * (camera_centers - minimum) / (maximum - minimum) - 1.0
        normalized_position = normalized_position.clamp(-1.0, 1.0)
        normalized_direction = F.normalize(view_directions, dim=-1, eps=1.0e-8)
        return torch.cat(
            (
                self.position_encoder(normalized_position),
                self.direction_encoder(normalized_direction),
            ),
            dim=-1,
        )


class GlobalImageAppearanceRefiner(nn.Module):
    """Image-wise 3x3 color affine transform from equations (5) and (6)."""

    def __init__(
        self,
        viewpoint_encoder: ViewpointEncoder,
        frame_embedding_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 4,
        matrix_delta_limit: float | None = None,
        bias_limit: float | None = None,
    ) -> None:
        super().__init__()
        if frame_embedding_dim <= 0:
            raise ValueError("frame_embedding_dim must be positive")
        if matrix_delta_limit is not None and matrix_delta_limit <= 0:
            raise ValueError("matrix_delta_limit must be positive")
        if bias_limit is not None and bias_limit <= 0:
            raise ValueError("bias_limit must be positive")
        self.viewpoint_encoder = viewpoint_encoder
        self.matrix_delta_limit = matrix_delta_limit
        self.bias_limit = bias_limit
        input_dim = viewpoint_encoder.output_dim + frame_embedding_dim
        self.affine_learner = MLP(input_dim, hidden_dim, 12, num_layers)
        self._initialize_identity()

    def _initialize_identity(self) -> None:
        final = self.affine_learner.final_layer
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.zero_()
            final.bias[:9].copy_(torch.eye(3).reshape(-1).to(final.bias))

    def predict_affine(
        self,
        frame_embeddings: Tensor,
        camera_centers: Tensor,
        view_directions: Tensor,
    ) -> GlobalAffine:
        if frame_embeddings.ndim == 1:
            frame_embeddings = frame_embeddings.unsqueeze(0)
        if camera_centers.ndim == 1:
            camera_centers = camera_centers.unsqueeze(0)
        if view_directions.ndim == 1:
            view_directions = view_directions.unsqueeze(0)
        batch_size = frame_embeddings.shape[0]
        if camera_centers.shape != (batch_size, 3):
            raise ValueError("camera_centers must have shape [B,3]")
        if view_directions.shape != (batch_size, 3):
            raise ValueError("view_directions must have shape [B,3]")

        viewpoint = self.viewpoint_encoder(camera_centers, view_directions)
        parameters = self.affine_learner(
            torch.cat((frame_embeddings.to(viewpoint), viewpoint), dim=-1)
        )
        matrix = parameters[..., :9].reshape(batch_size, 3, 3)
        bias = parameters[..., 9:]
        if self.matrix_delta_limit is not None:
            identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
            limit = self.matrix_delta_limit
            matrix = identity + limit * torch.tanh((matrix - identity) / limit)
        if self.bias_limit is not None:
            limit = self.bias_limit
            bias = limit * torch.tanh(bias / limit)
        return GlobalAffine(matrix=matrix, bias=bias)

    @staticmethod
    def apply_affine(images: Tensor, affine: GlobalAffine) -> Tensor:
        """Apply a batched affine to channel-last images.

        Accepted image shapes are ``[H,W,3]`` for a single affine or
        ``[B,...,3]`` for a batch. Values are deliberately not clamped during
        training.
        """

        if images.shape[-1] != 3:
            raise ValueError("images must use channel-last RGB")
        unbatched = images.ndim == 3
        if unbatched:
            if affine.matrix.shape[0] != 1:
                raise ValueError("an unbatched image requires exactly one affine")
            images = images.unsqueeze(0)
        if images.shape[0] != affine.matrix.shape[0]:
            raise ValueError("image and affine batch sizes must match")
        transformed = torch.einsum("bij,b...j->b...i", affine.matrix, images)
        bias_shape = (affine.bias.shape[0],) + (1,) * (images.ndim - 2) + (3,)
        transformed = transformed + affine.bias.reshape(bias_shape)
        return transformed[0] if unbatched else transformed

    def forward(
        self,
        images: Tensor,
        frame_embeddings: Tensor,
        camera_centers: Tensor,
        view_directions: Tensor,
    ) -> Tensor:
        affine = self.predict_affine(
            frame_embeddings, camera_centers, view_directions
        )
        return self.apply_affine(images, affine)

