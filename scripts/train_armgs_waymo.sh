#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_armgs_waymo.sh SEQUENCE START_FRAME END_FRAME [-- TRAINER_ARGS...]

Run one prepared Waymo sequence with the ArmGS paper-oriented defaults:
  - FRONT 1600x1066, StreetGS every-fourth holdout
  - all selected frames for LiDAR initialization, first LiDAR return
  - required train-only known-pose COLMAP points and Grounded-SAM sky masks
  - 30,000 steps in PAPER_MODE=1
  - W&B train GT/render every 500 steps
  - final-only reconstruction + novel-view PSNR/SSIM/LPIPS-Alex

Required prepared assets:
  <SKY_MASK_ROOT>/<SEQUENCE>/FRONT/<source-index:08d>.png
  <COLMAP_DIR>/triangulated_text/points3D.txt
  <COLMAP_DIR>/mapping.json

Environment overrides:
  WAYMO_ROOT       Waymo-v2 root (default: <ArmGS>/data/waymo_v2)
  PARQUET_DIR      split below WAYMO_ROOT (default: validation)
  PREPARED_ROOT    preprocessing root (default: <ArmGS>/data/waymo_prepared)
  CAS_TRACK_PATH   default: <PREPARED_ROOT>/tracking/castrack/<sequence>.json
                    mandatory and non-empty in paper mode
  ACTOR_BOX_SCALE  official planar scale from the sequence table in paper mode
  CACHE_DIR        canonical cache
  SKY_MASK_ROOT    sky-mask root
  COLMAP_DIR       known-pose COLMAP output
  COLMAP_POINTS3D  explicit points3D.txt
  OUTPUT_DIR       training output
  CONFIG           ArmGS YAML profile
  ARMGS_PYTHON     training Python (default: /venv/camosplat/bin/python)
  GPU_ID           physical GPU exposed as cuda:0 (optional)
  DEVICE           Torch device when GPU_ID is unset (default: cuda:0)
  PAPER_MODE       1 to enforce paper protocol, 0 for local smoke (default: 1)
  ITERATIONS       total steps (default: 30000; must be 30000 in paper mode)
  RESUME           checkpoint path
  CHECKPOINT_INTERVAL legacy compatibility setting; intermediate checkpoints are disabled
  LOG_INTERVAL     scalar interval (default: 100)
  IMAGE_LOG_INTERVAL
                    train GT/render interval (default: 500)
  EVAL_INTERVAL    periodic held-out evaluation (default: 0)
  WANDB_ENABLED    1/0 (default: 1)
  WANDB_ENTITY     default: CamoSplat_ICLR_2027
  WANDB_PROJECT    default: Ours-ArmGS-Waymo
  WANDB_RUN_NAME   default: armgs_waymo_<sequence>_paper
  WANDB_MODE       online/offline/disabled (default: online)
  WANDB_DIR        default: <ArmGS>/wandb
  DRY_RUN          1 prints the trainer command after preflight
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 3 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SEQUENCE="$1"
START_FRAME="$2"
END_FRAME="$3"
shift 3
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_TRAINER_ARGS=("$@")

[[ -n "${SEQUENCE}" && "${SEQUENCE}" != "." && "${SEQUENCE}" != ".." ]] ||
  die "SEQUENCE must be one non-empty context name"
[[ "${SEQUENCE}" != */* ]] || die "SEQUENCE cannot contain a slash"
[[ "${START_FRAME}" =~ ^[0-9]+$ ]] || die "START_FRAME must be non-negative"
[[ "${END_FRAME}" =~ ^[0-9]+$ ]] || die "END_FRAME must be non-negative"
(( END_FRAME >= START_FRAME )) || die "END_FRAME cannot be smaller than START_FRAME"

WAYMO_ROOT="${WAYMO_ROOT:-${ARMGS_ROOT}/data/waymo_v2}"
PARQUET_DIR="${PARQUET_DIR:-validation}"
PREPARED_ROOT="${PREPARED_ROOT:-${ARMGS_ROOT}/data/waymo_prepared}"
CAS_TRACK_PATH="${CAS_TRACK_PATH:-${PREPARED_ROOT}/tracking/castrack/${SEQUENCE}.json}"
ACTOR_BOX_SCALE="${ACTOR_BOX_SCALE:-}"
CACHE_DIR="${CACHE_DIR:-${PREPARED_ROOT}/cache}"
SKY_MASK_ROOT="${SKY_MASK_ROOT:-${PREPARED_ROOT}/sky_masks}"
COLMAP_DIR="${COLMAP_DIR:-${PREPARED_ROOT}/colmap/${SEQUENCE}}"
COLMAP_POINTS3D="${COLMAP_POINTS3D:-${COLMAP_DIR}/triangulated_text/points3D.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARMGS_ROOT}/outputs/waymo/${SEQUENCE}/paper}"
CONFIG="${CONFIG:-${ARMGS_ROOT}/configs/armgs_waymo_streetgs.yaml}"
ARMGS_PYTHON="${ARMGS_PYTHON:-/venv/camosplat/bin/python}"
PAPER_MODE="${PAPER_MODE:-1}"
ITERATIONS="${ITERATIONS:-30000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
IMAGE_LOG_INTERVAL="${IMAGE_LOG_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-CamoSplat_ICLR_2027}"
WANDB_PROJECT="${WANDB_PROJECT:-Ours-ArmGS-Waymo}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-armgs_waymo_${SEQUENCE}_paper}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-${ARMGS_ROOT}/wandb}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda:0}"
RESUME="${RESUME:-}"

for flag_name in PAPER_MODE WANDB_ENABLED DRY_RUN; do
  flag_value="${!flag_name}"
  [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] ||
    die "${flag_name} must be 0 or 1"
done
for integer_name in ITERATIONS CHECKPOINT_INTERVAL LOG_INTERVAL IMAGE_LOG_INTERVAL EVAL_INTERVAL; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] ||
    die "${integer_name} must be a non-negative integer"
done
(( ITERATIONS > 0 )) || die "ITERATIONS must be positive"
(( CHECKPOINT_INTERVAL > 0 )) || die "CHECKPOINT_INTERVAL must be positive"
(( LOG_INTERVAL > 0 )) || die "LOG_INTERVAL must be positive"

if [[ -n "${GPU_ID:-}" ]]; then
  [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  DEVICE="cuda:0"
fi

[[ -x "${ARMGS_PYTHON}" ]] || die "training Python is not executable: ${ARMGS_PYTHON}"
[[ -d "${WAYMO_ROOT}" ]] || die "Waymo root not found: ${WAYMO_ROOT}"
[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -f "${ARMGS_ROOT}/scripts/train_armgs_waymo.py" ]] ||
  die "Waymo trainer not found"

REQUIRED_COMPONENTS=(
  camera_image
  camera_calibration
  lidar
  lidar_pose
  lidar_box
  lidar_calibration
  vehicle_pose
)
MISSING_COMPONENTS=()
for component in "${REQUIRED_COMPONENTS[@]}"; do
  component_path="${WAYMO_ROOT}/${PARQUET_DIR}/${component}/${SEQUENCE}.parquet"
  [[ -s "${component_path}" ]] || MISSING_COMPONENTS+=("${component_path}")
done
if (( ${#MISSING_COMPONENTS[@]} > 0 )); then
  printf 'error: missing required Waymo-v2 component parquet(s):\n' >&2
  printf '  %s\n' "${MISSING_COMPONENTS[@]}" >&2
  exit 1
fi

SEQUENCE_TABLE="${ARMGS_ROOT}/configs/waymo_streetgs_sequences.txt"
OFFICIAL_ROW=""
if [[ -f "${SEQUENCE_TABLE}" ]]; then
  OFFICIAL_ROW="$(awk -v sequence="${SEQUENCE}" '
    $1 !~ /^#/ && $2 == sequence { print $1, $3, $4, $5; exit }
  ' "${SEQUENCE_TABLE}")"
fi
if [[ -n "${OFFICIAL_ROW}" ]]; then
  read -r OFFICIAL_SCENE OFFICIAL_START OFFICIAL_END OFFICIAL_BOX_SCALE <<< "${OFFICIAL_ROW}"
fi
if [[ "${PAPER_MODE}" == "1" ]]; then
  [[ "${PARQUET_DIR}" == "validation" ]] ||
    die "paper mode requires PARQUET_DIR=validation"
  [[ -n "${OFFICIAL_ROW}" ]] ||
    die "paper mode sequence is not one of configs/waymo_streetgs_sequences.txt"
  [[ "${START_FRAME}" == "${OFFICIAL_START}" &&
     "${END_FRAME}" == "${OFFICIAL_END}" ]] ||
    die "paper scene ${OFFICIAL_SCENE} requires frames ${OFFICIAL_START}..${OFFICIAL_END}"
  if [[ -n "${ACTOR_BOX_SCALE}" ]] &&
     ! awk -v actual="${ACTOR_BOX_SCALE}" -v expected="${OFFICIAL_BOX_SCALE}" \
       'BEGIN { delta = actual - expected; if (delta < 0) delta = -delta; exit !(delta <= 1e-12) }'; then
    die "paper scene ${OFFICIAL_SCENE} requires ACTOR_BOX_SCALE=${OFFICIAL_BOX_SCALE}"
  fi
  ACTOR_BOX_SCALE="${OFFICIAL_BOX_SCALE}"
  [[ "${ITERATIONS}" == "30000" ]] ||
    die "paper mode requires ITERATIONS=30000"
  [[ -s "${CAS_TRACK_PATH}" ]] ||
    die "paper mode requires non-empty CAStrack JSON: ${CAS_TRACK_PATH}"
elif [[ -z "${ACTOR_BOX_SCALE}" ]]; then
  ACTOR_BOX_SCALE="${OFFICIAL_BOX_SCALE:-1.0}"
fi
[[ "${ACTOR_BOX_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "ACTOR_BOX_SCALE must be a positive decimal"
awk -v value="${ACTOR_BOX_SCALE}" 'BEGIN { exit !(value > 0) }' ||
  die "ACTOR_BOX_SCALE must be positive"

[[ -s "${COLMAP_POINTS3D}" ]] ||
  die "COLMAP points are missing or empty: ${COLMAP_POINTS3D}"
[[ -s "${COLMAP_DIR}/mapping.json" ]] ||
  die "verified COLMAP mapping is missing: ${COLMAP_DIR}/mapping.json"

MISSING_MASKS=()
for (( source_index=START_FRAME; source_index<=END_FRAME; source_index++ )); do
  printf -v mask_name '%08d.png' "${source_index}"
  mask_path="${SKY_MASK_ROOT}/${SEQUENCE}/FRONT/${mask_name}"
  [[ -s "${mask_path}" ]] || MISSING_MASKS+=("${mask_path}")
  if (( ${#MISSING_MASKS[@]} >= 5 )); then
    break
  fi
done
if (( ${#MISSING_MASKS[@]} > 0 )); then
  printf 'error: sky-mask coverage is incomplete; first missing path(s):\n' >&2
  printf '  %s\n' "${MISSING_MASKS[@]}" >&2
  exit 1
fi

mkdir -p -- "${OUTPUT_DIR}" "${WANDB_DIR}"

TRAIN_ARGS=(
  "${ARMGS_PYTHON}"
  "${ARMGS_ROOT}/scripts/train_armgs_waymo.py"
  --config "${CONFIG}"
  --waymo-root "${WAYMO_ROOT}"
  --parquet-dir "${PARQUET_DIR}"
  --sequence "${SEQUENCE}"
  --start-frame "${START_FRAME}"
  --end-frame "${END_FRAME}"
  --cache-dir "${CACHE_DIR}"
  --sky-mask-root "${SKY_MASK_ROOT}"
  --colmap-points3d "${COLMAP_POINTS3D}"
  --actor-box-scale "${ACTOR_BOX_SCALE}"
  --camera FRONT
  --target-height 1066
  --target-width 1600
  --lidar-initialization-frames all-selected
  --lidar-returns first
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
  --iterations "${ITERATIONS}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --image-log-interval "${IMAGE_LOG_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-at-end
  --eval-reconstruction-at-end
  --eval-lpips
  --eval-lpips-net alex
)
if [[ -s "${CAS_TRACK_PATH}" ]]; then
  TRAIN_ARGS+=(--castrack-path "${CAS_TRACK_PATH}")
fi
if [[ "${PAPER_MODE}" == "1" ]]; then
  TRAIN_ARGS+=(--paper-mode)
fi
if [[ -n "${RESUME}" ]]; then
  [[ -s "${RESUME}" ]] || die "resume checkpoint not found: ${RESUME}"
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  TRAIN_ARGS+=(
    --wandb
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${WANDB_RUN_NAME}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-dir "${WANDB_DIR}"
  )
fi
TRAIN_ARGS+=("${EXTRA_TRAINER_ARGS[@]}")

printf 'ArmGS Waymo training preflight passed.\n'
printf '  sequence: %s (%s frames %s..%s)\n'   "${SEQUENCE}" "${PARQUET_DIR}" "${START_FRAME}" "${END_FRAME}"
printf '  actor tracking/planar box scale: %s / %s\n' \
  "${CAS_TRACK_PATH}" "${ACTOR_BOX_SCALE}"
printf '  initialization: all-selected LiDAR return1 + train-only known-pose COLMAP\n'
printf '  output: %s\n' "${OUTPUT_DIR}"
printf '  W&B: %s/%s (%s), image interval=%s\n'   "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_MODE}" "${IMAGE_LOG_INTERVAL}"
printf '  evaluation: periodic=%s, final reconstruction+novel PSNR/SSIM/LPIPS-Alex\n'   "${EVAL_INTERVAL}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'command:'
  printf ' %q' "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
fi

cd -- "${ARMGS_ROOT}"
exec "${TRAIN_ARGS[@]}"
