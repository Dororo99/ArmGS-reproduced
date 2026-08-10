#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_waymo_streetgs_scene.sh [--official] SCENE_ID
  scripts/prepare_waymo_streetgs_scene.sh --local-smoke [SEQUENCE [START END]]

Run the complete Waymo preparation/training pipeline:
  1. Grounded-SAM FRONT sky masks (camosplat decode -> armgs-gsam inference)
  2. train-only known-pose COLMAP triangulation with /usr/bin/colmap
  3. ArmGS training (30k paper mode for an official scene)

With no arguments the official scene 006 is selected. Official SCENE_ID must
exist in configs/waymo_streetgs_sequences.txt. The local smoke default is:
  12251442326766052580_1840_000_1860_000, training frames 0..15

Environment overrides:
  WAYMO_ROOT       default: <ArmGS>/data/waymo_v2
  PREPARED_ROOT    default: <ArmGS>/data/waymo_prepared
  CAS_TRACK_PATH   default: <PREPARED_ROOT>/tracking/castrack/<sequence>.json
  ACTOR_BOX_SCALE  official planar scale from the sequence table
  OUTPUT_DIR       training output
  GPU_ID           physical GPU (default: 0)
  RUN_SKY          1/0 (default: 1)
  RUN_COLMAP       1/0 (default: 1)
  RUN_TRAIN        1/0 (default: 1)
  REUSE_COLMAP     1 reuses a complete points3D+mapping pair (default: 1)
  COLMAP_USE_GPU   1/0 (default: 0)
  SAVE_OVERLAYS    sky QA overlays/contact sheet, 1/0 (default: 1)
  OVERWRITE        regenerate valid sky masks, 1/0 (default: 0)
  SMOKE_ITERATIONS local-smoke steps (default: 100)
  WANDB_ENABLED    1/0 (default: 1)
  WANDB_ENTITY     default: CamoSplat_ICLR_2027
  WANDB_PROJECT    default: Ours-ArmGS-Waymo
  WANDB_MODE       online/offline/disabled (default: online)
  DRY_RUN          1 prints all three commands after dataset preflight

Official mode deliberately fails before GPU work unless all seven validation
component parquets for the selected context are present.
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SEQUENCE_TABLE="${ARMGS_ROOT}/configs/waymo_streetgs_sequences.txt"

MODE="official"
SCENE_ID="006"
SEQUENCE=""
START_FRAME=""
END_FRAME=""

if (( $# > 0 )); then
  case "$1" in
    --official)
      MODE="official"
      [[ -n "${2:-}" ]] || die "--official requires SCENE_ID"
      SCENE_ID="$2"
      shift 2
      ;;
    --local-smoke)
      MODE="local-smoke"
      SEQUENCE="${2:-12251442326766052580_1840_000_1860_000}"
      START_FRAME="${3:-0}"
      END_FRAME="${4:-15}"
      if (( $# > 4 )); then
        die "--local-smoke accepts at most SEQUENCE START END"
      fi
      shift "$#"
      ;;
    *)
      MODE="official"
      SCENE_ID="$1"
      shift
      ;;
  esac
fi
(( $# == 0 )) || die "unexpected argument: $1"

if [[ "${MODE}" == "official" ]]; then
  [[ -f "${SEQUENCE_TABLE}" ]] || die "official sequence table not found"
  OFFICIAL_ROW="$(awk -v scene="${SCENE_ID}" '
    $1 !~ /^#/ && $1 == scene { print $1, $2, $3, $4, $5; exit }
  ' "${SEQUENCE_TABLE}")"
  [[ -n "${OFFICIAL_ROW}" ]] ||
    die "unknown official scene id: ${SCENE_ID}"
  read -r SCENE_ID SEQUENCE START_FRAME END_FRAME PLANAR_BOX_SCALE <<< "${OFFICIAL_ROW}"
  if [[ -n "${ACTOR_BOX_SCALE:-}" ]] &&
     ! awk -v actual="${ACTOR_BOX_SCALE}" -v expected="${PLANAR_BOX_SCALE}" \
       'BEGIN { delta = actual - expected; if (delta < 0) delta = -delta; exit !(delta <= 1e-12) }'; then
    die "official scene ${SCENE_ID} requires ACTOR_BOX_SCALE=${PLANAR_BOX_SCALE}"
  fi
  ACTOR_BOX_SCALE="${PLANAR_BOX_SCALE}"
  PARQUET_DIR="validation"
  PAPER_MODE="1"
  ITERATIONS="30000"
  RUN_TAG="scene_${SCENE_ID}"
else
  SCENE_ID="local"
  PARQUET_DIR="training"
  PAPER_MODE="0"
  ITERATIONS="${SMOKE_ITERATIONS:-100}"
  RUN_TAG="local_smoke"
  ACTOR_BOX_SCALE="${ACTOR_BOX_SCALE:-1.0}"
fi

[[ -n "${SEQUENCE}" && "${SEQUENCE}" != "." && "${SEQUENCE}" != ".." ]] ||
  die "SEQUENCE must be one non-empty context name"
[[ "${SEQUENCE}" != */* ]] || die "SEQUENCE cannot contain a slash"
[[ "${START_FRAME}" =~ ^[0-9]+$ ]] || die "START must be non-negative"
[[ "${END_FRAME}" =~ ^[0-9]+$ ]] || die "END must be non-negative"
(( END_FRAME >= START_FRAME )) || die "END cannot be smaller than START"

WAYMO_ROOT="${WAYMO_ROOT:-${ARMGS_ROOT}/data/waymo_v2}"
PREPARED_ROOT="${PREPARED_ROOT:-${ARMGS_ROOT}/data/waymo_prepared}"
CAS_TRACK_PATH="${CAS_TRACK_PATH:-${PREPARED_ROOT}/tracking/castrack/${SEQUENCE}.json}"
CACHE_DIR="${CACHE_DIR:-${PREPARED_ROOT}/cache}"
SKY_MASK_ROOT="${SKY_MASK_ROOT:-${PREPARED_ROOT}/sky_masks}"
FRAME_MANIFEST="${FRAME_MANIFEST:-${PREPARED_ROOT}/manifests/${SEQUENCE}_${START_FRAME}_${END_FRAME}.json}"
COLMAP_DIR="${COLMAP_DIR:-${PREPARED_ROOT}/colmap/${SEQUENCE}}"
COLMAP_POINTS3D="${COLMAP_POINTS3D:-${COLMAP_DIR}/triangulated_text/points3D.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARMGS_ROOT}/outputs/waymo/${RUN_TAG}}"
ARMGS_PYTHON="${ARMGS_PYTHON:-/venv/camosplat/bin/python}"
GSAM_PYTHON="${GSAM_PYTHON:-/venv/armgs-gsam/bin/python}"
COLMAP_BINARY="${COLMAP_BINARY:-/usr/bin/colmap}"
GPU_ID="${GPU_ID:-0}"
RUN_SKY="${RUN_SKY:-1}"
RUN_COLMAP="${RUN_COLMAP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
REUSE_COLMAP="${REUSE_COLMAP:-1}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-0}"
SAVE_OVERLAYS="${SAVE_OVERLAYS:-1}"
OVERWRITE="${OVERWRITE:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-CamoSplat_ICLR_2027}"
WANDB_PROJECT="${WANDB_PROJECT:-Ours-ArmGS-Waymo}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

for flag_name in RUN_SKY RUN_COLMAP RUN_TRAIN REUSE_COLMAP COLMAP_USE_GPU SAVE_OVERLAYS OVERWRITE WANDB_ENABLED DRY_RUN; do
  flag_value="${!flag_name}"
  [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] ||
    die "${flag_name} must be 0 or 1"
done
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "GPU_ID must be non-negative"
[[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "ITERATIONS must be positive"
[[ "${ACTOR_BOX_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "ACTOR_BOX_SCALE must be a positive decimal"
awk -v value="${ACTOR_BOX_SCALE}" 'BEGIN { exit !(value > 0) }' ||
  die "ACTOR_BOX_SCALE must be positive"

[[ -x "${ARMGS_PYTHON}" ]] ||
  die "camosplat Python is not executable: ${ARMGS_PYTHON}"
[[ -x "${GSAM_PYTHON}" ]] ||
  die "armgs-gsam Python is not executable: ${GSAM_PYTHON}"
[[ -x "${COLMAP_BINARY}" ]] ||
  die "system COLMAP is not executable: ${COLMAP_BINARY}"
[[ -x "${ARMGS_ROOT}/scripts/generate_waymo_sky_masks.sh" ]] ||
  die "sky-mask launcher is not executable"
[[ -f "${ARMGS_ROOT}/scripts/prepare_waymo_colmap.py" ]] ||
  die "Waymo COLMAP preparer is missing"
[[ -x "${ARMGS_ROOT}/scripts/train_armgs_waymo.sh" ]] ||
  die "Waymo train launcher is not executable"
[[ -d "${WAYMO_ROOT}" ]] || die "Waymo root not found: ${WAYMO_ROOT}"
if [[ "${PAPER_MODE}" == "1" ]]; then
  [[ -s "${CAS_TRACK_PATH}" ]] ||
    die "paper mode requires non-empty CAStrack JSON: ${CAS_TRACK_PATH}"
fi

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
  printf 'error: selected %s context is incomplete; missing component parquet(s):\n'     "${MODE}" >&2
  printf '  %s\n' "${MISSING_COMPONENTS[@]}" >&2
  if [[ "${MODE}" == "official" ]]; then
    printf 'The local training contexts are smoke-test data only; copy the official\n' >&2
    printf 'Waymo validation context into all seven component directories first.\n' >&2
  fi
  exit 1
fi

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

printf 'ArmGS Waymo pipeline\n'
printf '  mode: %s\n' "${MODE}"
printf '  scene/context: %s / %s\n' "${SCENE_ID}" "${SEQUENCE}"
printf '  split/range: %s, %s..%s inclusive\n'   "${PARQUET_DIR}" "${START_FRAME}" "${END_FRAME}"
printf '  actor tracking/planar box scale: %s / %s\n' \
  "${CAS_TRACK_PATH}" "${ACTOR_BOX_SCALE}"
printf '  environments: %s | %s | %s\n'   "${ARMGS_PYTHON}" "${GSAM_PYTHON}" "${COLMAP_BINARY}"
printf '  output: %s\n' "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p -- "${CACHE_DIR}" "${SKY_MASK_ROOT}"     "$(dirname -- "${FRAME_MANIFEST}")" "$(dirname -- "${COLMAP_DIR}")"
fi

if [[ "${RUN_SKY}" == "1" ]]; then
  printf '\n[1/3] Grounded-SAM sky masks\n'
  export PREPARE_PYTHON="${ARMGS_PYTHON}"
  export INFERENCE_PYTHON="${GSAM_PYTHON}"
  export WAYMO_ROOT PARQUET_DIR START_FRAME END_FRAME CACHE_DIR
  export OUTPUT_ROOT="${SKY_MASK_ROOT}"
  export FRAME_MANIFEST
  export SAVE_OVERLAYS OVERWRITE
  export PREPARE_ONLY=0
  run_or_print "${ARMGS_ROOT}/scripts/generate_waymo_sky_masks.sh"     "${SEQUENCE}" "${GPU_ID}"
else
  printf '\n[1/3] skipped (RUN_SKY=0)\n'
fi

if [[ "${RUN_COLMAP}" == "1" ]]; then
  printf '\n[2/3] train-only known-pose COLMAP\n'
  if [[ "${REUSE_COLMAP}" == "1" &&
        -s "${COLMAP_POINTS3D}" &&
        -s "${COLMAP_DIR}/mapping.json" ]]; then
    printf 'reuse complete COLMAP output: %s\n' "${COLMAP_POINTS3D}"
  else
    COLMAP_ARGS=(
      "${ARMGS_PYTHON}"
      "${ARMGS_ROOT}/scripts/prepare_waymo_colmap.py"
      --waymo-root "${WAYMO_ROOT}"
      --parquet-dir "${PARQUET_DIR}"
      --sequence "${SEQUENCE}"
      --cameras FRONT
      --start-frame "${START_FRAME}"
      --end-frame "${END_FRAME}"
      --target-height 1066
      --target-width 1600
      --cache-dir "${CACHE_DIR}"
      --output-dir "${COLMAP_DIR}"
      --split-every 4
      --split-offset 0
      --split-start-position 4
      --actor-box-scale "${ACTOR_BOX_SCALE}"
      --colmap-binary "${COLMAP_BINARY}"
    )
    if [[ -s "${CAS_TRACK_PATH}" ]]; then
      COLMAP_ARGS+=(--castrack-path "${CAS_TRACK_PATH}")
    fi
    if [[ "${COLMAP_USE_GPU}" == "1" ]]; then
      COLMAP_ARGS+=(--use-gpu)
    else
      COLMAP_ARGS+=(--no-use-gpu)
    fi
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    run_or_print "${COLMAP_ARGS[@]}"
  fi
else
  printf '\n[2/3] skipped (RUN_COLMAP=0)\n'
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  printf '\n[3/3] ArmGS training and final evaluation\n'
  export WAYMO_ROOT PARQUET_DIR PREPARED_ROOT CACHE_DIR SKY_MASK_ROOT
  export COLMAP_DIR COLMAP_POINTS3D OUTPUT_DIR ARMGS_PYTHON GPU_ID
  export CAS_TRACK_PATH ACTOR_BOX_SCALE
  export PAPER_MODE ITERATIONS WANDB_ENABLED WANDB_ENTITY WANDB_PROJECT WANDB_MODE
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-armgs_waymo_${RUN_TAG}}"
  export IMAGE_LOG_INTERVAL="${IMAGE_LOG_INTERVAL:-500}"
  export EVAL_INTERVAL=0
  run_or_print "${ARMGS_ROOT}/scripts/train_armgs_waymo.sh"     "${SEQUENCE}" "${START_FRAME}" "${END_FRAME}"
else
  printf '\n[3/3] skipped (RUN_TRAIN=0)\n'
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '\nDry-run complete; no preprocessing or training command was executed.\n'
else
  printf '\nPipeline complete.\n'
fi
