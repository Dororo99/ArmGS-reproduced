#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/generate_nuscenes_scene_0061_sky_masks.sh [GPU_ID]

Generate Grounded SAM sky masks for all six cameras in nuScenes scene-0061.
GPU_ID defaults to 0 and is mapped to cuda:0 through CUDA_VISIBLE_DEVICES.

Run scripts/setup_grounded_sam.sh first.

Optional environment overrides:
  CONDA_BIN       mamba/conda executable (auto-detected)
  ENV_NAME        Conda environment (default: armgs-gsam)
  NUSCENES_ROOT   Dataset root (default: <ArmGS>/data/nuscenes)
  NUSCENES_VERSION
                  Metadata version (default: v1.0-trainval)
  CAMERAS         all or comma-separated channels (default: all)
  OUTPUT_ROOT     Mask root before version/scene (default: data/sky_masks/nuscenes)
  GSAM_ROOT       Grounded SAM checkout directory
  CHECKPOINT_DIR  Grounded SAM checkpoint directory
  BOX_THRESHOLD   GroundingDINO box threshold (default: 0.3)
  TEXT_THRESHOLD  GroundingDINO text threshold (default: 0.25)
  NEGATIVE_TEXT_PROMPT
                  Exclusion prompt (default: building . tree; empty disables it)
  OVERLAY_EVERY   Save every Nth QA overlay (default: 1)
  SAVE_OVERLAYS   1 to save QA overlays, 0 to skip (default: 1)
  OVERWRITE       1 to replace existing masks, 0 to resume (default: 0)
  DRY_RUN         1 to enumerate inputs without inference (default: 0)

Output contract:
  <OUTPUT_ROOT>/<version>/scene-0061/<camera>/<sample_data_token>.png
  The grayscale PNG values are 255 for sky and 0 for non-sky.
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

ENV_NAME="${ENV_NAME:-armgs-gsam}"
NUSCENES_ROOT="${NUSCENES_ROOT:-${ARMGS_ROOT}/data/nuscenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
CAMERAS="${CAMERAS:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARMGS_ROOT}/data/sky_masks/nuscenes}"
GSAM_ROOT="${GSAM_ROOT:-${ARMGS_ROOT}/third_party/Grounded-Segment-Anything}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ARMGS_ROOT}/checkpoints/grounded_sam}"
GPU_ID="${1:-${GPU_ID:-0}}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
NEGATIVE_TEXT_PROMPT="${NEGATIVE_TEXT_PROMPT-building . tree}"
OVERLAY_EVERY="${OVERLAY_EVERY:-1}"
SAVE_OVERLAYS="${SAVE_OVERLAYS:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

GROUNDINGDINO_CONFIG="${GSAM_ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDINGDINO_CHECKPOINT="${CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"
BERT_DIR="${CHECKPOINT_DIR}/bert-base-uncased"
GSAM_HF_HOME="${CHECKPOINT_DIR}/huggingface"
SCENE="0061"
SCENE_OUTPUT_DIR="${OUTPUT_ROOT}/${NUSCENES_VERSION}/scene-${SCENE}"
CONTACT_SHEET="${CONTACT_SHEET:-${SCENE_OUTPUT_DIR}/sky_mask_contact_sheet.jpg}"

resolve_conda_bin() {
  local requested="${CONDA_BIN:-}"
  local candidate

  if [[ -n "${requested}" ]]; then
    if [[ -x "${requested}" ]]; then
      printf '%s\n' "${requested}"
      return
    fi
    if command -v "${requested}" >/dev/null 2>&1; then
      command -v "${requested}"
      return
    fi
    die "CONDA_BIN is not executable: ${requested}"
  fi
  for candidate in \
    /opt/miniforge3/bin/mamba \
    /opt/miniconda3/bin/mamba \
    /opt/miniforge3/bin/conda \
    /opt/miniconda3/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  for candidate in mamba conda; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  die "mamba/conda was not found; set CONDA_BIN explicitly"
}

absolute_from_root() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${ARMGS_ROOT}" "${path}"
  fi
}

CONDA_BIN="$(resolve_conda_bin)"
NUSCENES_ROOT="$(absolute_from_root "${NUSCENES_ROOT}")"
OUTPUT_ROOT="$(absolute_from_root "${OUTPUT_ROOT}")"
GSAM_ROOT="$(absolute_from_root "${GSAM_ROOT}")"
CHECKPOINT_DIR="$(absolute_from_root "${CHECKPOINT_DIR}")"

# Recompute dependent paths after normalizing optional relative overrides.
GROUNDINGDINO_CONFIG="${GSAM_ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDINGDINO_CHECKPOINT="${CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"
BERT_DIR="${CHECKPOINT_DIR}/bert-base-uncased"
GSAM_HF_HOME="${CHECKPOINT_DIR}/huggingface"
SCENE_OUTPUT_DIR="${OUTPUT_ROOT}/${NUSCENES_VERSION}/scene-${SCENE}"
CONTACT_SHEET="${CONTACT_SHEET:-${SCENE_OUTPUT_DIR}/sky_mask_contact_sheet.jpg}"
CONTACT_SHEET="$(absolute_from_root "${CONTACT_SHEET}")"

[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer"
[[ "${OVERLAY_EVERY}" =~ ^[1-9][0-9]*$ ]] || die "OVERLAY_EVERY must be positive"
for flag_name in SAVE_OVERLAYS OVERWRITE DRY_RUN; do
  flag_value="${!flag_name}"
  [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] || \
    die "${flag_name} must be 0 or 1"
done
[[ -d "${NUSCENES_ROOT}" ]] || die "nuScenes root not found: ${NUSCENES_ROOT}"
[[ -d "${NUSCENES_ROOT}/${NUSCENES_VERSION}" ]] || \
  die "nuScenes metadata version not found: ${NUSCENES_ROOT}/${NUSCENES_VERSION}"
[[ -f "${ARMGS_ROOT}/scripts/generate_nuscenes_sky_masks.py" ]] || \
  die "sky-mask generator not found: scripts/generate_nuscenes_sky_masks.py"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${GSAM_HF_HOME}"
export HF_HUB_CACHE="${GSAM_HF_HOME}/hub"
export TRANSFORMERS_CACHE="${GSAM_HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${GROUNDINGDINO_CONFIG}" ]] || \
    die "GroundingDINO config not found; run scripts/setup_grounded_sam.sh"
  [[ -s "${GROUNDINGDINO_CHECKPOINT}" ]] || \
    die "GroundingDINO checkpoint not found; run scripts/setup_grounded_sam.sh"
  [[ -s "${SAM_CHECKPOINT}" ]] || \
    die "SAM checkpoint not found; run scripts/setup_grounded_sam.sh"
  [[ -f "${BERT_DIR}/config.json" ]] || \
    die "BERT snapshot not found; run scripts/setup_grounded_sam.sh"

  "${CONDA_BIN}" run -n "${ENV_NAME}" python -c \
    'import torch; import groundingdino; import segment_anything; from groundingdino import _C; assert torch.cuda.is_available()' \
    >/dev/null

  mkdir -p -- "${OUTPUT_ROOT}"
fi

GENERATOR_ARGS=(
  "${CONDA_BIN}" run -n "${ENV_NAME}" python
  "${ARMGS_ROOT}/scripts/generate_nuscenes_sky_masks.py"
  --nuscenes-root "${NUSCENES_ROOT}"
  --version "${NUSCENES_VERSION}"
  --scene "${SCENE}"
  --cameras "${CAMERAS}"
  --output-root "${OUTPUT_ROOT}"
  --groundingdino-config "${GROUNDINGDINO_CONFIG}"
  --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}"
  --sam-checkpoint "${SAM_CHECKPOINT}"
  --sam-model-type vit_h
  --bert-path "${BERT_DIR}"
  --text-prompt sky
  --box-threshold "${BOX_THRESHOLD}"
  --text-threshold "${TEXT_THRESHOLD}"
  --device cuda:0
)

if [[ -n "${NEGATIVE_TEXT_PROMPT}" ]]; then
  GENERATOR_ARGS+=(--negative-text-prompt "${NEGATIVE_TEXT_PROMPT}")
fi

if [[ "${SAVE_OVERLAYS}" == "1" ]]; then
  GENERATOR_ARGS+=(
    --save-overlays
    --overlay-every "${OVERLAY_EVERY}"
    --contact-sheet "${CONTACT_SHEET}"
  )
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  GENERATOR_ARGS+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  GENERATOR_ARGS+=(--dry-run)
fi

printf 'Grounded SAM sky-mask generation\n'
printf '  scene: scene-%s (%s, cameras=%s)\n' \
  "${SCENE}" "${NUSCENES_VERSION}" "${CAMERAS}"
printf '  GPU: physical %s -> cuda:0\n' "${GPU_ID}"
printf '  output: %s\n' "${SCENE_OUTPUT_DIR}"
printf '  thresholds: box=%s text=%s\n' "${BOX_THRESHOLD}" "${TEXT_THRESHOLD}"
printf '  prompts: sky; exclude=%s\n' "${NEGATIVE_TEXT_PROMPT:-<disabled>}"

cd -- "${ARMGS_ROOT}"
"${GENERATOR_ARGS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '\nDry-run complete; no masks or QA files were written.\n'
else
  printf '\nGeneration complete.\n'
  printf '  manifest: %s/generation_manifest.json\n' "${SCENE_OUTPUT_DIR}"
  if [[ "${SAVE_OVERLAYS}" == "1" ]]; then
    printf '  contact sheet: %s\n' "${CONTACT_SHEET}"
  fi
fi
