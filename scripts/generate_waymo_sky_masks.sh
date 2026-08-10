#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/generate_waymo_sky_masks.sh SEQUENCE [GPU_ID]

Generate Waymo FRONT sky masks through two isolated Python environments:
  1. camosplat: decode Waymo parquet and atomically prepare a frame manifest.
  2. armgs-gsam: validate that manifest and run Grounded-SAM without Waymo SDK.

GPU_ID defaults to 0 and is mapped to cuda:0 through CUDA_VISIBLE_DEVICES.

Optional environment overrides:
  PREPARE_PYTHON  direct decoder Python (default: /venv/camosplat/bin/python)
  INFERENCE_PYTHON
                  direct GSAM Python (default: /venv/armgs-gsam/bin/python)
  CONDA_BIN       mamba/conda executable; fallback if direct Python is absent
  PREPARE_ENV     Waymo decoding environment (default: camosplat)
  INFERENCE_ENV   Grounded-SAM environment (default: armgs-gsam)
  WAYMO_ROOT      Waymo v2 root (default: /workspace/data/waymo_v2)
  PARQUET_DIR     split below WAYMO_ROOT (default: validation)
  START_FRAME     inclusive source index (default: 0)
  END_FRAME       inclusive source index (default: all available frames)
  CACHE_DIR       decoded image cache (default: <ArmGS>/data/waymo_cache)
  FRAME_MANIFEST  stage boundary JSON (default: below CACHE_DIR/sky_manifests)
  OUTPUT_ROOT     mask root (default: <ArmGS>/data/sky_masks/waymo)
  GSAM_ROOT       Grounded SAM checkout
  CHECKPOINT_DIR  Grounded SAM checkpoints
  BOX_THRESHOLD   GroundingDINO box threshold (default: 0.3)
  TEXT_THRESHOLD  GroundingDINO text threshold (default: 0.25)
  OVERLAY_EVERY   save every Nth QA overlay (default: 1)
  SAVE_OVERLAYS   1 to save overlays/contact sheet (default: 1)
  OVERWRITE       1 to regenerate valid masks (default: 0)
  DRY_RUN         1 to validate prepared inputs without model inference (default: 0)
  PREPARE_ONLY    1 to stop after the camosplat stage (default: 0)

Output contract:
  <OUTPUT_ROOT>/<SEQUENCE>/FRONT/<absolute-source-frame-index:08d>.png
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
if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SEQUENCE="$1"
GPU_ID="${2:-${GPU_ID:-0}}"

PREPARE_PYTHON="${PREPARE_PYTHON:-/venv/camosplat/bin/python}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-/venv/armgs-gsam/bin/python}"
PREPARE_ENV="${PREPARE_ENV:-camosplat}"
INFERENCE_ENV="${INFERENCE_ENV:-armgs-gsam}"
WAYMO_ROOT="${WAYMO_ROOT:-/workspace/data/waymo_v2}"
PARQUET_DIR="${PARQUET_DIR:-validation}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-}"
CACHE_DIR="${CACHE_DIR:-${ARMGS_ROOT}/data/waymo_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARMGS_ROOT}/data/sky_masks/waymo}"
GSAM_ROOT="${GSAM_ROOT:-${ARMGS_ROOT}/third_party/Grounded-Segment-Anything}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ARMGS_ROOT}/checkpoints/grounded_sam}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
OVERLAY_EVERY="${OVERLAY_EVERY:-1}"
SAVE_OVERLAYS="${SAVE_OVERLAYS:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
TARGET_HEIGHT="${TARGET_HEIGHT:-1066}"
TARGET_WIDTH="${TARGET_WIDTH:-1600}"
SOURCE_IMAGE_HEIGHT="${SOURCE_IMAGE_HEIGHT:-1280}"
TOP_EDGE_ORIGINAL_PIXELS="${TOP_EDGE_ORIGINAL_PIXELS:-100}"

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

[[ -n "${SEQUENCE}" && "${SEQUENCE}" != "." && "${SEQUENCE}" != ".." ]] || \
  die "SEQUENCE must be one non-empty context name"
[[ "${SEQUENCE}" != */* ]] || die "SEQUENCE cannot contain a slash"
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer"
[[ "${START_FRAME}" =~ ^[0-9]+$ ]] || die "START_FRAME must be non-negative"
if [[ -n "${END_FRAME}" ]]; then
  [[ "${END_FRAME}" =~ ^[0-9]+$ ]] || die "END_FRAME must be non-negative"
  (( END_FRAME >= START_FRAME )) || die "END_FRAME cannot be smaller than START_FRAME"
fi
for positive_name in OVERLAY_EVERY TARGET_HEIGHT TARGET_WIDTH SOURCE_IMAGE_HEIGHT TOP_EDGE_ORIGINAL_PIXELS; do
  positive_value="${!positive_name}"
  [[ "${positive_value}" =~ ^[1-9][0-9]*$ ]] || die "${positive_name} must be positive"
done
for flag_name in SAVE_OVERLAYS OVERWRITE DRY_RUN PREPARE_ONLY; do
  flag_value="${!flag_name}"
  [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] || \
    die "${flag_name} must be 0 or 1"
done

FALLBACK_CONDA=""
if [[ ! -x "${PREPARE_PYTHON}" || ! -x "${INFERENCE_PYTHON}" ]]; then
  FALLBACK_CONDA="$(resolve_conda_bin)"
fi
if [[ -x "${PREPARE_PYTHON}" ]]; then
  PREPARE_COMMAND=("${PREPARE_PYTHON}")
  PREPARE_RUNTIME="${PREPARE_PYTHON}"
else
  PREPARE_COMMAND=("${FALLBACK_CONDA}" run -n "${PREPARE_ENV}" python)
  PREPARE_RUNTIME="conda:${PREPARE_ENV}"
fi
if [[ -x "${INFERENCE_PYTHON}" ]]; then
  INFERENCE_COMMAND=("${INFERENCE_PYTHON}")
  INFERENCE_RUNTIME="${INFERENCE_PYTHON}"
else
  INFERENCE_COMMAND=("${FALLBACK_CONDA}" run -n "${INFERENCE_ENV}" python)
  INFERENCE_RUNTIME="conda:${INFERENCE_ENV}"
fi
WAYMO_ROOT="$(absolute_from_root "${WAYMO_ROOT}")"
CACHE_DIR="$(absolute_from_root "${CACHE_DIR}")"
OUTPUT_ROOT="$(absolute_from_root "${OUTPUT_ROOT}")"
GSAM_ROOT="$(absolute_from_root "${GSAM_ROOT}")"
CHECKPOINT_DIR="$(absolute_from_root "${CHECKPOINT_DIR}")"
FRAME_TAG="${START_FRAME}_${END_FRAME:-all}"
FRAME_MANIFEST="${FRAME_MANIFEST:-${CACHE_DIR}/sky_manifests/${PARQUET_DIR}/${SEQUENCE}_${FRAME_TAG}.json}"
FRAME_MANIFEST="$(absolute_from_root "${FRAME_MANIFEST}")"

GENERATOR="${ARMGS_ROOT}/scripts/generate_waymo_sky_masks.py"
GROUNDINGDINO_CONFIG="${GSAM_ROOT}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDINGDINO_CHECKPOINT="${CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"
BERT_DIR="${CHECKPOINT_DIR}/bert-base-uncased"
GSAM_HF_HOME="${CHECKPOINT_DIR}/huggingface"
SEQUENCE_OUTPUT="${OUTPUT_ROOT}/${SEQUENCE}"
CONTACT_SHEET="${CONTACT_SHEET:-${SEQUENCE_OUTPUT}/sky_mask_contact_sheet.jpg}"
CONTACT_SHEET="$(absolute_from_root "${CONTACT_SHEET}")"

[[ -d "${WAYMO_ROOT}" ]] || die "Waymo root not found: ${WAYMO_ROOT}"
[[ -d "${WAYMO_ROOT}/${PARQUET_DIR}" ]] || \
  die "Waymo parquet split not found: ${WAYMO_ROOT}/${PARQUET_DIR}"
[[ -f "${GENERATOR}" ]] || die "generator not found: ${GENERATOR}"

COMMON_ARGS=(
  --waymo-root "${WAYMO_ROOT}"
  --parquet-dir "${PARQUET_DIR}"
  --sequence "${SEQUENCE}"
  --start-frame "${START_FRAME}"
  --target-height "${TARGET_HEIGHT}"
  --target-width "${TARGET_WIDTH}"
  --cache-dir "${CACHE_DIR}"
  --output-root "${OUTPUT_ROOT}"
)
if [[ -n "${END_FRAME}" ]]; then
  COMMON_ARGS+=(--end-frame "${END_FRAME}")
fi

mkdir -p -- "${CACHE_DIR}" "${OUTPUT_ROOT}"

printf 'Waymo sky-mask stage 1/2: decode and prepare\n'
printf '  python: %s\n' "${PREPARE_RUNTIME}"
printf '  sequence: %s (%s, frames %s..%s)\n' \
  "${SEQUENCE}" "${PARQUET_DIR}" "${START_FRAME}" "${END_FRAME:-end}"
printf '  frame manifest: %s\n' "${FRAME_MANIFEST}"

cd -- "${ARMGS_ROOT}"
"${PREPARE_COMMAND[@]}" "${GENERATOR}" \
  "${COMMON_ARGS[@]}" --prepare-manifest "${FRAME_MANIFEST}"
[[ -s "${FRAME_MANIFEST}" ]] || \
  die "prepare stage did not create the frame manifest: ${FRAME_MANIFEST}"

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  printf '\nPrepare stage complete.\n'
  exit 0
fi

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${GROUNDINGDINO_CONFIG}" ]] || \
    die "GroundingDINO config not found; run scripts/setup_grounded_sam.sh"
  [[ -s "${GROUNDINGDINO_CHECKPOINT}" ]] || \
    die "GroundingDINO checkpoint not found; run scripts/setup_grounded_sam.sh"
  [[ -s "${SAM_CHECKPOINT}" ]] || \
    die "SAM checkpoint not found; run scripts/setup_grounded_sam.sh"
  [[ -f "${BERT_DIR}/config.json" ]] || \
    die "BERT snapshot not found; run scripts/setup_grounded_sam.sh"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${GSAM_HF_HOME}"
export HF_HUB_CACHE="${GSAM_HF_HOME}/hub"
export TRANSFORMERS_CACHE="${GSAM_HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

INFERENCE_ARGS=(
  "${INFERENCE_COMMAND[@]}" "${GENERATOR}"
  "${COMMON_ARGS[@]}"
  --input-frame-manifest "${FRAME_MANIFEST}"
  --groundingdino-config "${GROUNDINGDINO_CONFIG}"
  --groundingdino-checkpoint "${GROUNDINGDINO_CHECKPOINT}"
  --sam-checkpoint "${SAM_CHECKPOINT}"
  --sam-model-type vit_h
  --bert-path "${BERT_DIR}"
  --text-prompt sky
  --box-threshold "${BOX_THRESHOLD}"
  --text-threshold "${TEXT_THRESHOLD}"
  --device cuda:0
  --source-image-height "${SOURCE_IMAGE_HEIGHT}"
  --top-edge-original-pixels "${TOP_EDGE_ORIGINAL_PIXELS}"
)
if [[ "${SAVE_OVERLAYS}" == "1" ]]; then
  INFERENCE_ARGS+=(
    --save-overlays
    --overlay-every "${OVERLAY_EVERY}"
    --contact-sheet "${CONTACT_SHEET}"
  )
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  INFERENCE_ARGS+=(--overwrite)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  INFERENCE_ARGS+=(--dry-run)
fi

printf '\nWaymo sky-mask stage 2/2: Grounded-SAM inference\n'
printf '  python: %s\n' "${INFERENCE_RUNTIME}"
printf '  GPU: physical %s -> cuda:0\n' "${GPU_ID}"
printf '  output: %s\n' "${SEQUENCE_OUTPUT}"
printf '  thresholds: box=%s text=%s\n' "${BOX_THRESHOLD}" "${TEXT_THRESHOLD}"
"${INFERENCE_ARGS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '\nDry-run complete; prepared inputs were validated and no masks were written.\n'
else
  printf '\nGeneration complete.\n'
  printf '  manifest: %s/generation_manifest.json\n' "${SEQUENCE_OUTPUT}"
  if [[ "${SAVE_OVERLAYS}" == "1" ]]; then
    printf '  contact sheet: %s\n' "${CONTACT_SHEET}"
  fi
fi
