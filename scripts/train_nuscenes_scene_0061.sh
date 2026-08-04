#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_nuscenes_scene_0061.sh [GPU_ID]

Train ArmGS on nuScenes v1.0-trainval scene-0061 using all six cameras.
GPU_ID defaults to 0 and is mapped to cuda:0 through CUDA_VISIBLE_DEVICES.

Optional environment overrides:
  PYTHON_BIN          Python executable (default: /venv/camosplat/bin/python)
  NUSCENES_ROOT       Dataset root (default: <ArmGS>/data/nuscenes)
  NUSCENES_VERSION    Metadata version (default: v1.0-trainval)
  SKY_MASK_ROOT       Sky masks (default: <ArmGS>/data/sky_masks/nuscenes/<version>/scene-0061)
  SKY_MASK_REJECT_LIST Reviewed token list (default: configs/nuscenes_scene_0061_sky_mask_reject_tokens.txt; empty disables)
  COLMAP_POINTS3D     Optional known-pose, nuScenes-world points3D.txt
  OUTPUT_DIR          Run output directory
  RESUME              Checkpoint path to resume
  ITERATIONS          Total training iterations (default: 30000)
  CHECKPOINT_INTERVAL Checkpoint interval (default: 1000)
  LOG_INTERVAL        Training scalar/terminal log interval (default: 100)
  IMAGE_LOG_INTERVAL  W&B training GT/render image interval; 0 disables (default: 500)
  EVAL_INTERVAL       Held-out PSNR/SSIM/actor-PSNR interval; 0 disables periodic (default: 1000)
  EVAL_AT_END         Run final held-out evaluation, 0 or 1 (default: 1)
  EVAL_LPIPS          Also compute held-out LPIPS, 0 or 1 (default: 1)
  EVAL_LPIPS_NET      LPIPS backbone: alex, vgg, or squeeze (default: alex)
  EVAL_ONLY           Evaluate RESUME without training/checkpoint writes (default: 0)
  WANDB_ENTITY        W&B entity/team (default: CamoSplat)
  WANDB_PROJECT       W&B project (default: ArmGS-nuScenes)
  WANDB_RUN_NAME      W&B run name
  WANDB_RUN_ID        Explicit run-ID override; otherwise OUTPUT_DIR sidecar auto-resumes
  WANDB_MODE          W&B mode (default: online)
  WANDB_DIR           Local W&B directory (default: <OUTPUT_DIR>/wandb)
  WANDB_FAIL_FAST     Stop training on W&B errors, 0 or 1 (default: 0)
  WANDB_LOG_CHECKPOINT Upload only the final checkpoint as an Artifact, 0 or 1 (default: 0)

Example:
  scripts/train_nuscenes_scene_0061.sh 0
  RESUME=/path/to/checkpoint.pt OUTPUT_DIR=/path/to/existing/run \
    scripts/train_nuscenes_scene_0061.sh 1
  EVAL_ONLY=1 RESUME=/path/to/final.pt OUTPUT_DIR=/path/to/existing/run \
    scripts/train_nuscenes_scene_0061.sh 0
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
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

PYTHON_BIN="${PYTHON_BIN:-/venv/camosplat/bin/python}"
NUSCENES_ROOT_INPUT="${NUSCENES_ROOT:-${ARMGS_ROOT}/data/nuscenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
SCENE="0061"
SKY_MASK_ROOT_INPUT="${SKY_MASK_ROOT:-${ARMGS_ROOT}/data/sky_masks/nuscenes/${NUSCENES_VERSION}/scene-${SCENE}}"
SKY_MASK_REJECT_LIST_INPUT="${SKY_MASK_REJECT_LIST-${ARMGS_ROOT}/configs/nuscenes_scene_0061_sky_mask_reject_tokens.txt}"
COLMAP_POINTS3D_INPUT="${COLMAP_POINTS3D:-}"
CAMERAS="all"
GPU_ID="${1:-${GPU_ID:-0}}"
ITERATIONS="${ITERATIONS:-30000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
IMAGE_LOG_INTERVAL="${IMAGE_LOG_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_AT_END="${EVAL_AT_END:-1}"
EVAL_LPIPS="${EVAL_LPIPS:-1}"
EVAL_LPIPS_NET="${EVAL_LPIPS_NET:-alex}"
EVAL_ONLY="${EVAL_ONLY:-0}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARMGS_ROOT}/outputs/nuscenes/scene_0061/${RUN_TIMESTAMP}}"
RESUME="${RESUME:-}"

WANDB_ENTITY="${WANDB_ENTITY:-CamoSplat}"
WANDB_PROJECT="${WANDB_PROJECT:-ArmGS-nuScenes}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-armgs_nuscenes_scene_0061_${RUN_TIMESTAMP}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_FAIL_FAST="${WANDB_FAIL_FAST:-0}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-0}"

[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer, got '${GPU_ID}'"
[[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || die "ITERATIONS must be positive"
[[ "${CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "CHECKPOINT_INTERVAL must be positive"
[[ "${LOG_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || die "LOG_INTERVAL must be positive"
[[ "${IMAGE_LOG_INTERVAL}" =~ ^[0-9]+$ ]] || die "IMAGE_LOG_INTERVAL must be non-negative"
[[ "${EVAL_INTERVAL}" =~ ^[0-9]+$ ]] || die "EVAL_INTERVAL must be non-negative"
[[ "${EVAL_AT_END}" =~ ^[01]$ ]] || die "EVAL_AT_END must be 0 or 1"
[[ "${EVAL_LPIPS}" =~ ^[01]$ ]] || die "EVAL_LPIPS must be 0 or 1"
[[ "${EVAL_LPIPS_NET}" =~ ^(alex|vgg|squeeze)$ ]] || die "EVAL_LPIPS_NET must be alex, vgg, or squeeze"
[[ "${EVAL_ONLY}" =~ ^[01]$ ]] || die "EVAL_ONLY must be 0 or 1"
[[ "${WANDB_FAIL_FAST}" =~ ^[01]$ ]] || die "WANDB_FAIL_FAST must be 0 or 1"
[[ "${WANDB_LOG_CHECKPOINT}" =~ ^[01]$ ]] || die "WANDB_LOG_CHECKPOINT must be 0 or 1"
if [[ "${EVAL_ONLY}" == "1" && -z "${RESUME}" ]]; then
  die "EVAL_ONLY=1 requires RESUME"
fi
[[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ -f "${ARMGS_ROOT}/scripts/train_armgs_nuscenes.py" ]] || \
  die "nuScenes trainer not found: ${ARMGS_ROOT}/scripts/train_armgs_nuscenes.py"
[[ -f "${ARMGS_ROOT}/configs/armgs_nuscenes_scene_0061.yaml" ]] || \
  die "scene configuration not found"
[[ -d "${NUSCENES_ROOT_INPUT}" ]] || die "nuScenes root not found: ${NUSCENES_ROOT_INPUT}"

NUSCENES_ROOT="$(cd -- "${NUSCENES_ROOT_INPUT}" && pwd -P)"
if [[ "${SKY_MASK_ROOT_INPUT}" != /* ]]; then
  SKY_MASK_ROOT_INPUT="${ARMGS_ROOT}/${SKY_MASK_ROOT_INPUT}"
fi
[[ -d "${SKY_MASK_ROOT_INPUT}" ]] || \
  die "sky mask root not found: ${SKY_MASK_ROOT_INPUT}"
SKY_MASK_ROOT="$(cd -- "${SKY_MASK_ROOT_INPUT}" && pwd -P)"
SKY_MASK_REJECT_LIST_PATH=""
if [[ -n "${SKY_MASK_REJECT_LIST_INPUT}" ]]; then
  if [[ "${SKY_MASK_REJECT_LIST_INPUT}" != /* ]]; then
    SKY_MASK_REJECT_LIST_INPUT="${ARMGS_ROOT}/${SKY_MASK_REJECT_LIST_INPUT}"
  fi
  [[ -f "${SKY_MASK_REJECT_LIST_INPUT}" ]] || \
    die "sky mask reject list not found: ${SKY_MASK_REJECT_LIST_INPUT}"
  SKY_MASK_REJECT_LIST_DIR="$(cd -- "$(dirname -- "${SKY_MASK_REJECT_LIST_INPUT}")" && pwd -P)"
  SKY_MASK_REJECT_LIST_PATH="${SKY_MASK_REJECT_LIST_DIR}/$(basename -- "${SKY_MASK_REJECT_LIST_INPUT}")"
fi
COLMAP_POINTS3D_PATH=""
if [[ -n "${COLMAP_POINTS3D_INPUT}" ]]; then
  if [[ "${COLMAP_POINTS3D_INPUT}" != /* ]]; then
    COLMAP_POINTS3D_INPUT="${ARMGS_ROOT}/${COLMAP_POINTS3D_INPUT}"
  fi
  [[ -f "${COLMAP_POINTS3D_INPUT}" ]] || \
    die "COLMAP points3D file not found: ${COLMAP_POINTS3D_INPUT}"
  COLMAP_POINTS3D_DIR="$(cd -- "$(dirname -- "${COLMAP_POINTS3D_INPUT}")" && pwd -P)"
  COLMAP_POINTS3D_PATH="${COLMAP_POINTS3D_DIR}/$(basename -- "${COLMAP_POINTS3D_INPUT}")"
fi
VERSION_DIR="${NUSCENES_ROOT}/${NUSCENES_VERSION}"
[[ -d "${VERSION_DIR}" ]] || die "nuScenes metadata version not found: ${VERSION_DIR}"

REQUIRED_TABLES=(
  scene.json
  sample.json
  sample_data.json
  sensor.json
  calibrated_sensor.json
  ego_pose.json
  sample_annotation.json
  instance.json
  category.json
)
for table in "${REQUIRED_TABLES[@]}"; do
  [[ -f "${VERSION_DIR}/${table}" ]] || die "missing nuScenes table: ${VERSION_DIR}/${table}"
done

REQUIRED_SENSORS=(
  CAM_FRONT
  CAM_FRONT_LEFT
  CAM_FRONT_RIGHT
  CAM_BACK
  CAM_BACK_LEFT
  CAM_BACK_RIGHT
  LIDAR_TOP
)
for sensor in "${REQUIRED_SENSORS[@]}"; do
  [[ -d "${NUSCENES_ROOT}/samples/${sensor}" ]] || \
    die "missing nuScenes sample directory: ${NUSCENES_ROOT}/samples/${sensor}"
done

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ARMGS_ROOT}/${OUTPUT_DIR}"
fi
mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd -P)"

WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
if [[ "${WANDB_DIR}" != /* ]]; then
  WANDB_DIR="${ARMGS_ROOT}/${WANDB_DIR}"
fi
mkdir -p -- "${WANDB_DIR}"
if [[ "${EVAL_LPIPS}" == "1" ]]; then
  "${PYTHON_BIN}" -c 'import lpips'
fi

WANDB_DIR="$(cd -- "${WANDB_DIR}" && pwd -P)"

if [[ -n "${RESUME}" ]]; then
  if [[ "${RESUME}" != /* ]]; then
    RESUME="${ARMGS_ROOT}/${RESUME}"
  fi
  [[ -f "${RESUME}" ]] || die "resume checkpoint not found: ${RESUME}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ARMGS_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_ENTITY
export WANDB_PROJECT
export WANDB_RUN_NAME
export WANDB_NAME="${WANDB_NAME:-${WANDB_RUN_NAME}}"
export WANDB_MODE
export WANDB_DIR

"${PYTHON_BIN}" -c \
  'import torch, wandb, yaml; import gsplat; assert torch.cuda.is_available(), "CUDA is unavailable"; assert torch.cuda.device_count() > 0, "selected GPU is unavailable"'

"${PYTHON_BIN}" -c \
  'import json, pathlib, sys; rows=json.loads(pathlib.Path(sys.argv[1]).read_text()); name=sys.argv[2]; assert any(row.get("name") == name for row in rows), f"{name} is absent from scene.json"' \
  "${VERSION_DIR}/scene.json" "scene-${SCENE}"

TRAIN_ARGS=(
  "${PYTHON_BIN}"
  "${ARMGS_ROOT}/scripts/train_armgs_nuscenes.py"
  --config "${ARMGS_ROOT}/configs/armgs_nuscenes_scene_0061.yaml"
  --nuscenes-root "${NUSCENES_ROOT}"
  --sky-mask-root "${SKY_MASK_ROOT}"
  --scene "${SCENE}"
  --version "${NUSCENES_VERSION}"
  --cameras "${CAMERAS}"
  --device cuda:0
  --output-dir "${OUTPUT_DIR}"
  --iterations "${ITERATIONS}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --image-log-interval "${IMAGE_LOG_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-lpips-net "${EVAL_LPIPS_NET}"
  --wandb
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-dir "${WANDB_DIR}"
)
if [[ -n "${WANDB_RUN_ID}" ]]; then
  TRAIN_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
fi
if [[ "${WANDB_FAIL_FAST}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb-fail-fast)
else
  TRAIN_ARGS+=(--no-wandb-fail-fast)
fi
if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb-log-checkpoint-artifact)
else
  TRAIN_ARGS+=(--no-wandb-log-checkpoint-artifact)
fi
if [[ "${EVAL_AT_END}" == "1" ]]; then
  TRAIN_ARGS+=(--eval-at-end)
else
  TRAIN_ARGS+=(--no-eval-at-end)
fi
if [[ "${EVAL_LPIPS}" == "1" ]]; then
  TRAIN_ARGS+=(--eval-lpips)
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  TRAIN_ARGS+=(--eval-only)
fi
if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${SKY_MASK_REJECT_LIST_PATH}" ]]; then
  TRAIN_ARGS+=(--sky-mask-reject-list "${SKY_MASK_REJECT_LIST_PATH}")
fi
if [[ -n "${COLMAP_POINTS3D_PATH}" ]]; then
  TRAIN_ARGS+=(--colmap-points3d "${COLMAP_POINTS3D_PATH}")
fi

printf 'ArmGS nuScenes scene-%s training\n' "${SCENE}"
printf '  dataset: %s (%s)\n' "${NUSCENES_ROOT}" "${NUSCENES_VERSION}"
printf '  sky masks: %s\n' "${SKY_MASK_ROOT}"
if [[ -n "${COLMAP_POINTS3D_PATH}" ]]; then
  printf '  SfM points: %s\n' "${COLMAP_POINTS3D_PATH}"
else
  printf '  warning: COLMAP_POINTS3D is unset; initialization is LiDAR-only\n' >&2
fi
printf '  sky reject list: %s\n' "${SKY_MASK_REJECT_LIST_PATH:-disabled}"
printf '  GPU: physical %s -> cuda:0\n' "${GPU_ID}"
printf '  output: %s\n' "${OUTPUT_DIR}"
printf '  logging: scalars=%s train_images=%s checkpoints=%s\n' \
  "${LOG_INTERVAL}" "${IMAGE_LOG_INTERVAL}" "${CHECKPOINT_INTERVAL}"
printf '  evaluation: interval=%s at_end=%s lpips=%s/%s eval_only=%s\n' \
  "${EVAL_INTERVAL}" "${EVAL_AT_END}" "${EVAL_LPIPS}" "${EVAL_LPIPS_NET}" "${EVAL_ONLY}"
printf '  W&B: %s/%s/%s (%s, id=%s, fail_fast=%s, final_artifact=%s)\n' \
  "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_RUN_NAME}" "${WANDB_MODE}" \
  "${WANDB_RUN_ID:-auto}" "${WANDB_FAIL_FAST}" "${WANDB_LOG_CHECKPOINT}"

cd -- "${ARMGS_ROOT}"
if [[ "${EVAL_ONLY}" == "1" ]]; then
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/evaluation.log"
else
  "${TRAIN_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/training.log"
fi
