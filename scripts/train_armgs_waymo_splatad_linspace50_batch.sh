#!/usr/bin/env bash
set -Eeuo pipefail

# Train the ten Waymo contexts with SplatAD's official per-sensor LINSPACE
# split. For N FRONT captures, training uses:
#   numpy.linspace(0, N - 1, ceil(N * 0.5), dtype=int64)
# and validation uses the sorted complement. LiDAR initialization and known-
# pose COLMAP are restricted to those training captures to avoid holdout
# sensor/RGB leakage.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# These values define this launcher and are intentionally not overridable.
# Every scene must create/resume its own sidecar-managed W&B identity.
unset WANDB_RUN_ID WANDB_RESUME
export SPLIT_TYPE=linspace
export TRAIN_SPLIT_FRACTION=0.5
export LIDAR_INITIALIZATION_FRAMES=train-only

# Keep every protocol-dependent artifact separate from the older StreetGS
# every-fourth experiment. Sky masks, decoded caches, and actor tracks remain
# shared because they do not depend on the train/evaluation split.
export COLMAP_TAG=colmap_splatad_linspace50
export OUTPUT_TAG=splatad_linspace50_30k
export LOG_ROOT="${LOG_ROOT:-${ARMGS_ROOT}/logs/waymo_splatad_linspace50_batch}"
export RUN_NAME_PREFIX=armgs_waymo_splatad_linspace50

# The delegated batch keeps its existing operational controls:
#   GPU_IDS=0,1                         choose two GPUs
#   --only SEQUENCE                     run one statically assigned scene
#   --dry-run                           inspect without writing
#   --no-prepare                        require existing LINSPACE50 COLMAP
#   WAIT_FOR_FREE_GPU=1                 wait instead of oversubscribing
#   WANDB_ENTITY / WANDB_PROJECT        select the W&B destination
#
# Examples:
#   GPU_IDS=0,1 scripts/train_armgs_waymo_splatad_linspace50_batch.sh
#   GPU_IDS=7,6 scripts/train_armgs_waymo_splatad_linspace50_batch.sh \
#     --only 4986495627634617319_2980_000_3000_000

exec "${ARMGS_ROOT}/scripts/train_armgs_waymo_splatad_batch.sh" "$@"
