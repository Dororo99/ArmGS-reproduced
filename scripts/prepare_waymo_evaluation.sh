#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_waymo_evaluation.sh SEQUENCE [OUTPUT_DIR]

Validate the seven Waymo-v2 parquet components and prepare FRONT-camera RGB
targets plus reconstruction/novel-view evaluation manifests. This does not
create an ArmGS training manifest or initialize Gaussians.

Environment overrides:
  PYTHON_BIN          Python executable (default: /venv/camosplat/bin/python)
  WAYMO_ROOT          Waymo-v2 root (default: /workspace/data/waymo_v2)
  PARQUET_DIR         Component subset directory (default: training)
  CAMERAS             Comma-separated channels (default: FRONT)
  START_FRAME         First source capture, inclusive (default: 0)
  END_FRAME           Last source capture, inclusive (default: all)
  TEST_EVERY          Held-out capture interval (default: 4)
  FIRST_TEST_POSITION First held-out position relative to selected range (default: 4)
  TARGET_HEIGHT       Extracted image height (default: 1066)
  TARGET_WIDTH        Extracted image width (default: 1600)
  EXTRACT_IMAGES      Write lossless PNG targets/manifests, 0 or 1 (default: 1)

START_FRAME and END_FRAME use the same inclusive semantics as the official
StreetGS selected_frames setting. See configs/waymo_streetgs_sequences.txt.

Examples:
  EXTRACT_IMAGES=0 scripts/prepare_waymo_evaluation.sh \
    12251442326766052580_1840_000_1860_000

  PARQUET_DIR=validation START_FRAME=0 END_FRAME=85 \
    scripts/prepare_waymo_evaluation.sh \
    10448102132863604198_472_000_492_000
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
PYTHON_BIN="${PYTHON_BIN:-/venv/camosplat/bin/python}"
WAYMO_ROOT_INPUT="${WAYMO_ROOT:-/workspace/data/waymo_v2}"
PARQUET_DIR="${PARQUET_DIR:-training}"
CAMERAS="${CAMERAS:-FRONT}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-}"
TEST_EVERY="${TEST_EVERY:-4}"
FIRST_TEST_POSITION="${FIRST_TEST_POSITION:-4}"
TARGET_HEIGHT="${TARGET_HEIGHT:-1066}"
TARGET_WIDTH="${TARGET_WIDTH:-1600}"
EXTRACT_IMAGES="${EXTRACT_IMAGES:-1}"
OUTPUT_DIR="${2:-${ARMGS_ROOT}/data/waymo_prepared/${SEQUENCE}}"

[[ "${SEQUENCE}" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid sequence name: ${SEQUENCE}"
[[ "${PARQUET_DIR}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "PARQUET_DIR must be one directory name"
[[ "${START_FRAME}" =~ ^[0-9]+$ ]] || die "START_FRAME must be non-negative"
[[ "${TEST_EVERY}" =~ ^[1-9][0-9]*$ ]] || die "TEST_EVERY must be positive"
[[ "${FIRST_TEST_POSITION}" =~ ^[0-9]+$ ]] || die "FIRST_TEST_POSITION must be non-negative"
[[ "${TARGET_HEIGHT}" =~ ^[1-9][0-9]*$ ]] || die "TARGET_HEIGHT must be positive"
[[ "${TARGET_WIDTH}" =~ ^[1-9][0-9]*$ ]] || die "TARGET_WIDTH must be positive"
[[ "${EXTRACT_IMAGES}" =~ ^[01]$ ]] || die "EXTRACT_IMAGES must be 0 or 1"
if [[ -n "${END_FRAME}" ]]; then
  [[ "${END_FRAME}" =~ ^[0-9]+$ ]] || die "END_FRAME must be non-negative"
  (( 10#${END_FRAME} >= 10#${START_FRAME} )) || \
    die "END_FRAME must be greater than or equal to START_FRAME"
fi

[[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ -f "${ARMGS_ROOT}/scripts/prepare_waymo_v2_evaluation.py" ]] || \
  die "Waymo preparation script not found"
[[ -d "${WAYMO_ROOT_INPUT}" ]] || die "Waymo root not found: ${WAYMO_ROOT_INPUT}"
WAYMO_ROOT="$(cd -- "${WAYMO_ROOT_INPUT}" && pwd -P)"

REQUIRED_COMPONENTS=(
  camera_image
  camera_calibration
  lidar
  lidar_pose
  lidar_box
  lidar_calibration
  vehicle_pose
)
for component in "${REQUIRED_COMPONENTS[@]}"; do
  component_path="${WAYMO_ROOT}/${PARQUET_DIR}/${component}/${SEQUENCE}.parquet"
  [[ -f "${component_path}" ]] || die "missing Waymo component: ${component_path}"
done

"${PYTHON_BIN}" -c 'import pyarrow.parquet'
if [[ "${EXTRACT_IMAGES}" == "1" ]]; then
  "${PYTHON_BIN}" -c 'from PIL import Image'
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ARMGS_ROOT}/${OUTPUT_DIR}"
fi

PREPARE_ARGS=(
  "${PYTHON_BIN}"
  "${ARMGS_ROOT}/scripts/prepare_waymo_v2_evaluation.py"
  --waymo-root "${WAYMO_ROOT}"
  --parquet-dir "${PARQUET_DIR}"
  --sequence "${SEQUENCE}"
  --cameras "${CAMERAS}"
  --start-frame "${START_FRAME}"
  --test-every "${TEST_EVERY}"
  --first-test-position "${FIRST_TEST_POSITION}"
  --target-height "${TARGET_HEIGHT}"
  --target-width "${TARGET_WIDTH}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${END_FRAME}" ]]; then
  PREPARE_ARGS+=(--end-frame "${END_FRAME}")
fi
if [[ "${EXTRACT_IMAGES}" == "1" ]]; then
  PREPARE_ARGS+=(--extract-images)
else
  PREPARE_ARGS+=(--no-extract-images)
fi

printf 'ArmGS Waymo evaluation preparation\n'
printf '  sequence: %s\n' "${SEQUENCE}"
printf '  source: %s/%s\n' "${WAYMO_ROOT}" "${PARQUET_DIR}"
printf '  cameras: %s\n' "${CAMERAS}"
printf '  range: [%s, %s] (inclusive)\n' "${START_FRAME}" "${END_FRAME:-end}"
printf '  held out: relative positions %s, %s+%s, ...\n' \
  "${FIRST_TEST_POSITION}" "${FIRST_TEST_POSITION}" "${TEST_EVERY}"
printf '  output: %s\n' "${OUTPUT_DIR}"
printf '  image extraction: %s\n' "${EXTRACT_IMAGES}"

cd -- "${ARMGS_ROOT}"
"${PREPARE_ARGS[@]}"
