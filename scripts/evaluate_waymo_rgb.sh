#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_waymo_rgb.sh MANIFEST [OUTPUT_JSON]

Compute mean RGB PSNR, SSIM, and LPIPS-Alex from a prepared Waymo manifest.
The renderer must first write every prediction path listed in the manifest.

Environment override:
  PYTHON_BIN  Python executable (default: /venv/camosplat/bin/python)
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
PYTHON_BIN="${PYTHON_BIN:-/venv/camosplat/bin/python}"
MANIFEST="$1"
OUTPUT_JSON="${2:-${MANIFEST%.json}_metrics.json}"

[[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"
[[ -f "${ARMGS_ROOT}/scripts/evaluate_armgs.py" ]] || die "evaluator not found"
"${PYTHON_BIN}" -c 'import lpips, torch; from PIL import Image'

"${PYTHON_BIN}" "${ARMGS_ROOT}/scripts/evaluate_armgs.py" \
  --manifest "${MANIFEST}" \
  --data-range 1.0 \
  --ssim-window-size 11 \
  --ssim-sigma 1.5 \
  --lpips \
  --lpips-net alex \
  --output "${OUTPUT_JSON}"
