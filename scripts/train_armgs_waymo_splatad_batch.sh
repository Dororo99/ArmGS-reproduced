#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_armgs_waymo_splatad_batch.sh [OPTIONS]

Prepare and train the ten local SplatAD Waymo contexts on two GPUs. The two
GPU workers run concurrently, while each worker processes its five scenes
strictly sequentially:

  GPU_IDS[0]: scene indices 0, 2, 4, 6, 8
  GPU_IDS[1]: scene indices 1, 3, 5, 7, 9

Every scene runs:
  1. Grounded-SAM FRONT sky masks when missing
  2. train-only centered known-pose COLMAP when missing
  3. 30k ArmGS training and final reconstruction/novel-view evaluation

Options:
  --dry-run          Print queues and commands without reading data or writing
  --only SEQUENCE    Run one sequence on its statically assigned GPU
  --no-prepare       Require existing sky masks and COLMAP outputs
  -h, --help         Show this help

Environment overrides:
  GPU_IDS                 Two comma-separated physical GPUs (default: 0,1)
  WAYMO_ROOT              Waymo-v2 root (default: <ArmGS>/data/waymo_v2)
  PREPARED_ROOT           Shared preparation root
  OUTPUT_ROOT             Per-context output parent
  LOG_ROOT                Per-context launcher log parent
  SPLIT_TYPE              streetgs-periodic or linspace
  TRAIN_SPLIT_FRACTION    fraction for linspace (default: 0.5)
  LIDAR_INITIALIZATION_FRAMES
                           all-selected or train-only
  COLMAP_TAG              COLMAP asset directory name (default: colmap)
  RUN_NAME_PREFIX         W&B run-name prefix
  RUN_PREPARE             Generate/reuse sky masks and COLMAP, 1/0 (default: 1)
  BATCH_LOCK              Reject a concurrent copy of this batch, 1/0 (default: 1)
  AUTO_RESUME             Resume the newest checkpoint, 1/0 (default: 1)
  SKIP_COMPLETED          Skip checkpoint+final-eval complete scenes (default: 1)
  CONTINUE_ON_ERROR       Continue the same GPU queue after a failure (default: 1)
  WAIT_FOR_FREE_GPU       Wait while a GPU is occupied, 1/0 (default: 1)
  GPU_BUSY_THRESHOLD_MIB  Memory at or below this is considered free (default: 64)
  GPU_POLL_SECONDS        Busy-GPU polling interval (default: 30)
  ITERATIONS              Training steps (default: 30000)
  CHECKPOINT_INTERVAL     Legacy compatibility setting; intermediate checkpoints are disabled
  LOG_INTERVAL            Scalar/W&B interval (default: 100)
  IMAGE_LOG_INTERVAL      Train GT/render interval (default: 500)
  WANDB_ENABLED           W&B logging, 1/0 (default: 1)
  WANDB_ENTITY            default: CamoSplat_ICLR_2027
  WANDB_PROJECT           default: Ours-ArmGS-Waymo
  WANDB_MODE              online/offline/disabled (default: online)
  COLMAP_USE_GPU          COLMAP SIFT GPU use, 1/0 (default: 0)
  SAVE_OVERLAYS           Save sky-mask QA overlays, 1/0 (default: 0)

Protocol note:
  These are Waymo training contexts, not the eight ArmGS/StreetGS validation
  contexts. They intentionally use PAPER_MODE=0, full context, FRONT only,
  every-fourth held-out frames, ACTOR_BOX_SCALE=1.0, and Waymo GT lidar_box
  actor tracks because CAStrack results are unavailable for these contexts.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DRY_RUN="${DRY_RUN:-0}"
ONLY_SCENE="${ONLY_SCENE:-}"
RUN_PREPARE="${RUN_PREPARE:-1}"

while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --only)
      [[ -n "${2:-}" ]] || die "--only requires a sequence"
      ONLY_SCENE="$2"
      shift 2
      ;;
    --no-prepare)
      RUN_PREPARE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

GPU_IDS_RAW="${GPU_IDS:-0,1}"
WAYMO_ROOT="${WAYMO_ROOT:-${ARMGS_ROOT}/data/waymo_v2}"
PREPARED_ROOT="${PREPARED_ROOT:-${ARMGS_ROOT}/data/waymo_prepared}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARMGS_ROOT}/outputs/waymo}"
LOG_ROOT="${LOG_ROOT:-${ARMGS_ROOT}/logs/waymo_splatad_batch}"
OUTPUT_TAG="${OUTPUT_TAG:-splatad_30k}"
SPLIT_TYPE="${SPLIT_TYPE:-streetgs-periodic}"
TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION:-0.5}"
LIDAR_INITIALIZATION_FRAMES="${LIDAR_INITIALIZATION_FRAMES:-all-selected}"
COLMAP_TAG="${COLMAP_TAG:-colmap}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-armgs_waymo_splatad}"
ARMGS_PYTHON="${ARMGS_PYTHON:-/venv/camosplat/bin/python}"
GSAM_PYTHON="${GSAM_PYTHON:-/venv/armgs-gsam/bin/python}"
COLMAP_BINARY="${COLMAP_BINARY:-/usr/bin/colmap}"
CONFIG="${CONFIG:-${ARMGS_ROOT}/configs/armgs_waymo_streetgs.yaml}"
WANDB_DIR="${WANDB_DIR:-${ARMGS_ROOT}/wandb}"

AUTO_RESUME="${AUTO_RESUME:-1}"
BATCH_LOCK="${BATCH_LOCK:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-0}"
SAVE_OVERLAYS="${SAVE_OVERLAYS:-0}"
ITERATIONS="${ITERATIONS:-30000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
IMAGE_LOG_INTERVAL="${IMAGE_LOG_INTERVAL:-500}"
GPU_BUSY_THRESHOLD_MIB="${GPU_BUSY_THRESHOLD_MIB:-64}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
WANDB_ENTITY="${WANDB_ENTITY:-CamoSplat_ICLR_2027}"
WANDB_PROJECT="${WANDB_PROJECT:-Ours-ArmGS-Waymo}"
WANDB_MODE="${WANDB_MODE:-online}"

PREPARE_LAUNCHER="${ARMGS_ROOT}/scripts/prepare_waymo_streetgs_scene.sh"
TRAIN_LAUNCHER="${ARMGS_ROOT}/scripts/train_armgs_waymo.sh"

# Entries are sequence:start:end. The ranges are the complete timestamp-sorted
# contexts and use the inclusive indexing contract of the ArmGS Waymo loader.
SCENES=(
  "4986495627634617319_2980_000_3000_000:0:198"
  "4672649953433758614_2700_000_2720_000:0:198"
  "6791933003490312185_2607_000_2627_000:0:197"
  "17364342162691622478_780_000_800_000:0:198"
  "3385534893506316900_4252_000_4272_000:0:197"
  "9747453753779078631_940_000_960_000:0:197"
  "14940138913070850675_5755_330_5775_330:0:196"
  "204421859195625800_1080_000_1100_000:0:197"
  "7566697458525030390_1440_000_1460_000:0:197"
  "17159836069183024120_640_000_660_000:0:198"
)

for flag_name in DRY_RUN RUN_PREPARE BATCH_LOCK AUTO_RESUME SKIP_COMPLETED \
  CONTINUE_ON_ERROR WAIT_FOR_FREE_GPU WANDB_ENABLED COLMAP_USE_GPU SAVE_OVERLAYS; do
  flag_value="${!flag_name}"
  [[ "${flag_value}" == "0" || "${flag_value}" == "1" ]] ||
    die "${flag_name} must be 0 or 1"
done
for integer_name in ITERATIONS CHECKPOINT_INTERVAL LOG_INTERVAL \
  IMAGE_LOG_INTERVAL GPU_BUSY_THRESHOLD_MIB GPU_POLL_SECONDS; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] ||
    die "${integer_name} must be a non-negative integer"
done
(( ITERATIONS > 0 )) || die "ITERATIONS must be positive"
(( CHECKPOINT_INTERVAL > 0 )) || die "CHECKPOINT_INTERVAL must be positive"
(( LOG_INTERVAL > 0 )) || die "LOG_INTERVAL must be positive"
(( GPU_POLL_SECONDS > 0 )) || die "GPU_POLL_SECONDS must be positive"
[[ "${OUTPUT_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "OUTPUT_TAG must be one safe path component"
[[ "${COLMAP_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "COLMAP_TAG must be one safe path component"
[[ "${RUN_NAME_PREFIX}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
  die "RUN_NAME_PREFIX must be one safe W&B name component"
case "${SPLIT_TYPE}" in
  streetgs-periodic|linspace) ;;
  *) die "SPLIT_TYPE must be streetgs-periodic or linspace" ;;
esac
awk -v value="${TRAIN_SPLIT_FRACTION}"   'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value < 1) }' ||
  die "TRAIN_SPLIT_FRACTION must satisfy 0 < value < 1"
case "${LIDAR_INITIALIZATION_FRAMES}" in
  all-selected|train-only) ;;
  *) die "LIDAR_INITIALIZATION_FRAMES must be all-selected or train-only" ;;
esac

IFS=',' read -r -a GPU_ID_VALUES <<< "${GPU_IDS_RAW}"
(( ${#GPU_ID_VALUES[@]} == 2 )) ||
  die "GPU_IDS must contain exactly two comma-separated GPU ids"
for gpu_id in "${GPU_ID_VALUES[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu_id}"
done
[[ "${GPU_ID_VALUES[0]}" != "${GPU_ID_VALUES[1]}" ]] ||
  die "GPU_IDS must contain two distinct GPUs"

GPU0_QUEUE=()
GPU1_QUEUE=()
matched_only=0
for index in "${!SCENES[@]}"; do
  entry="${SCENES[index]}"
  IFS=':' read -r sequence _ _ <<< "${entry}"
  if [[ -n "${ONLY_SCENE}" && "${sequence}" != "${ONLY_SCENE}" ]]; then
    continue
  fi
  matched_only=1
  if (( index % 2 == 0 )); then
    GPU0_QUEUE+=("${entry}")
  else
    GPU1_QUEUE+=("${entry}")
  fi
done
if [[ -n "${ONLY_SCENE}" && "${matched_only}" == "0" ]]; then
  die "--only sequence is not in the ten-scene list: ${ONLY_SCENE}"
fi

print_queue() {
  local gpu="$1"
  shift
  local entry sequence start end
  printf 'GPU %s queue (%s scene(s)):\n' "${gpu}" "$#"
  for entry in "$@"; do
    IFS=':' read -r sequence start end <<< "${entry}"
    printf '  %s  frames=%s..%s\n' "${sequence}" "${start}" "${end}"
  done
}

print_header() {
  printf 'ArmGS Waymo two-GPU batch\n'
  printf '  protocol: training split, full context, FRONT, %s (fraction=%s)\n' \
    "${SPLIT_TYPE}" "${TRAIN_SPLIT_FRACTION}"
  printf '  LiDAR initialization: %s\n' "${LIDAR_INITIALIZATION_FRAMES}"
  printf '  actors: Waymo GT lidar_box fallback, planar box scale=1.0\n'
  printf '  steps/preparation: %s / %s\n' "${ITERATIONS}" "${RUN_PREPARE}"
  printf '  outputs: %s/<sequence>/%s\n' "${OUTPUT_ROOT}" "${OUTPUT_TAG}"
  printf '  W&B: %s/%s (%s), images every %s steps\n' \
    "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_MODE}" \
    "${IMAGE_LOG_INTERVAL}"
  print_queue "${GPU_ID_VALUES[0]}" "${GPU0_QUEUE[@]}"
  print_queue "${GPU_ID_VALUES[1]}" "${GPU1_QUEUE[@]}"
}

print_dry_run_scene() {
  local gpu="$1"
  local entry="$2"
  local sequence start end output_dir
  IFS=':' read -r sequence start end <<< "${entry}"
  output_dir="${OUTPUT_ROOT}/${sequence}/${OUTPUT_TAG}"
  if [[ "${RUN_PREPARE}" == "1" ]]; then
    printf '[dry-run][GPU %s] prepare %s (%s..%s)\n' \
      "${gpu}" "${sequence}" "${start}" "${end}"
  fi
  printf '[dry-run][GPU %s] train %s -> %s\n' \
    "${gpu}" "${sequence}" "${output_dir}"
  printf '  env: PARQUET_DIR=training PAPER_MODE=0 ITERATIONS=%s ACTOR_BOX_SCALE=1.0 SPLIT_TYPE=%s TRAIN_SPLIT_FRACTION=%s\n' \
    "${ITERATIONS}" "${SPLIT_TYPE}" "${TRAIN_SPLIT_FRACTION}"
}

print_header
if [[ "${DRY_RUN}" == "1" ]]; then
  for entry in "${GPU0_QUEUE[@]}"; do
    print_dry_run_scene "${GPU_ID_VALUES[0]}" "${entry}"
  done
  for entry in "${GPU1_QUEUE[@]}"; do
    print_dry_run_scene "${GPU_ID_VALUES[1]}" "${entry}"
  done
  printf 'Dry-run complete; no files were read or written.\n'
  exit 0
fi

[[ -d "${WAYMO_ROOT}" ]] || die "Waymo root not found: ${WAYMO_ROOT}"
[[ -x "${ARMGS_PYTHON}" ]] || die "ArmGS Python is not executable: ${ARMGS_PYTHON}"
[[ -x "${TRAIN_LAUNCHER}" ]] || die "training launcher is not executable"
[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
if [[ "${RUN_PREPARE}" == "1" ]]; then
  [[ -x "${PREPARE_LAUNCHER}" ]] || die "preparation launcher is not executable"
  [[ -x "${GSAM_PYTHON}" ]] || die "Grounded-SAM Python is not executable: ${GSAM_PYTHON}"
  [[ -x "${COLMAP_BINARY}" ]] || die "COLMAP is not executable: ${COLMAP_BINARY}"
fi
if [[ "${WAIT_FOR_FREE_GPU}" == "1" ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required to wait for GPUs"
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
for entry in "${GPU0_QUEUE[@]}" "${GPU1_QUEUE[@]}"; do
  IFS=':' read -r sequence _ _ <<< "${entry}"
  for component in "${REQUIRED_COMPONENTS[@]}"; do
    component_path="${WAYMO_ROOT}/training/${component}/${sequence}.parquet"
    [[ -s "${component_path}" ]] || die "missing Waymo component: ${component_path}"
  done
done

mkdir -p -- "${PREPARED_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}" "${WANDB_DIR}"
if [[ "${BATCH_LOCK}" == "1" ]]; then
  command -v flock >/dev/null || die "flock is required when BATCH_LOCK=1"
  exec 9>"${LOG_ROOT}/.batch.lock"
  flock -n 9 || die "another Waymo SplatAD batch is already running"
fi

final_step_name() {
  printf 'step_%08d' "${ITERATIONS}"
}

final_checkpoint_name() {
  printf "final.pt"
}

scene_assets_ready() {
  local sequence="$1"
  local start="$2"
  local end="$3"
  local colmap_dir mask_name source_index
  colmap_dir="${PREPARED_ROOT}/${COLMAP_TAG}/${sequence}"
  [[ -s "${colmap_dir}/triangulated_text/points3D.txt" ]] || return 1
  [[ -s "${colmap_dir}/mapping.json" ]] || return 1
  for (( source_index=start; source_index<=end; source_index++ )); do
    printf -v mask_name '%08d.png' "${source_index}"
    [[ -s "${PREPARED_ROOT}/sky_masks/${sequence}/FRONT/${mask_name}" ]] ||
      return 1
  done
}

latest_checkpoint() {
  local output_dir="$1"
  local checkpoint latest=""
  local -a checkpoints=()
  if [[ -d "${output_dir}/checkpoints" ]]; then
    shopt -s nullglob
    checkpoints=("${output_dir}"/checkpoints/step_*.pt)
    shopt -u nullglob
    for checkpoint in "${checkpoints[@]}"; do
      [[ -s "${checkpoint}" ]] && latest="${checkpoint}"
    done
  fi
  printf '%s' "${latest}"
}

scene_complete() {
  local output_dir="$1"
  local step_name
  step_name="$(final_step_name)"
  [[ -s "${output_dir}/checkpoints/$(final_checkpoint_name)" ]] &&
    [[ -s "${output_dir}/evaluation/novel_view/${step_name}.json" ]] &&
    [[ -s "${output_dir}/evaluation/reconstruction/${step_name}.json" ]]
}

wait_for_gpu() {
  local gpu="$1"
  local output used_mib
  [[ "${WAIT_FOR_FREE_GPU}" == "1" ]] || return 0
  while true; do
    if ! output="$(nvidia-smi -i "${gpu}" \
      --query-compute-apps=used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null)"; then
      printf 'error: failed to query GPU %s\n' "${gpu}" >&2
      return 1
    fi
    used_mib="$(awk '$1 ~ /^[0-9]+$/ { sum += $1 } END { print sum + 0 }' <<< "${output}")"
    if (( used_mib <= GPU_BUSY_THRESHOLD_MIB )); then
      return 0
    fi
    printf '[GPU %s] busy (%s MiB); waiting %s seconds\n' \
      "${gpu}" "${used_mib}" "${GPU_POLL_SECONDS}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

prepare_scene() {
  local gpu="$1"
  local sequence="$2"
  local start="$3"
  local end="$4"
  local output_dir="$5"
  local colmap_dir castrack_path
  colmap_dir="${PREPARED_ROOT}/${COLMAP_TAG}/${sequence}"
  castrack_path="${PREPARED_ROOT}/tracking/castrack/${sequence}.json"

  if scene_assets_ready "${sequence}" "${start}" "${end}"; then
    printf '[GPU %s][%s] prepared sky/COLMAP assets already complete\n' \
      "${gpu}" "${sequence}"
    return 0
  fi
  [[ "${RUN_PREPARE}" == "1" ]] || {
    printf 'error: [%s] sky masks or COLMAP output are missing; remove --no-prepare\n' \
      "${sequence}" >&2
    return 1
  }

  printf '[GPU %s][%s] preparing sky masks and known-pose COLMAP\n' \
    "${gpu}" "${sequence}"
  env \
    WAYMO_ROOT="${WAYMO_ROOT}" \
    PREPARED_ROOT="${PREPARED_ROOT}" \
    CACHE_DIR="${PREPARED_ROOT}/cache" \
    SKY_MASK_ROOT="${PREPARED_ROOT}/sky_masks" \
    FRAME_MANIFEST="${PREPARED_ROOT}/manifests/${sequence}_${start}_${end}.json" \
    COLMAP_DIR="${colmap_dir}" \
    COLMAP_POINTS3D="${colmap_dir}/triangulated_text/points3D.txt" \
    CAS_TRACK_PATH="${castrack_path}" \
    ACTOR_BOX_SCALE=1.0 \
    SPLIT_TYPE="${SPLIT_TYPE}" \
    TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION}" \
    LIDAR_INITIALIZATION_FRAMES="${LIDAR_INITIALIZATION_FRAMES}" \
    OUTPUT_DIR="${output_dir}" \
    ARMGS_PYTHON="${ARMGS_PYTHON}" \
    GSAM_PYTHON="${GSAM_PYTHON}" \
    COLMAP_BINARY="${COLMAP_BINARY}" \
    GPU_ID="${gpu}" \
    RUN_SKY=1 \
    RUN_COLMAP=1 \
    RUN_TRAIN=0 \
    REUSE_COLMAP=1 \
    COLMAP_USE_GPU="${COLMAP_USE_GPU}" \
    SAVE_OVERLAYS="${SAVE_OVERLAYS}" \
    OVERWRITE=0 \
    WANDB_ENABLED=0 \
    "${PREPARE_LAUNCHER}" --local-smoke "${sequence}" "${start}" "${end}"
}

train_scene() {
  local gpu="$1"
  local sequence="$2"
  local start="$3"
  local end="$4"
  local output_dir="$5"
  local resume="$6"
  local eval_only="$7"
  local colmap_dir castrack_path run_name
  local -a extra_args=()
  colmap_dir="${PREPARED_ROOT}/${COLMAP_TAG}/${sequence}"
  castrack_path="${PREPARED_ROOT}/tracking/castrack/${sequence}.json"
  run_name="${RUN_NAME_PREFIX}_${sequence}_${ITERATIONS}"
  if [[ "${eval_only}" == "1" ]]; then
    extra_args=(-- --eval-only)
  fi

  env \
    WAYMO_ROOT="${WAYMO_ROOT}" \
    PARQUET_DIR=training \
    PREPARED_ROOT="${PREPARED_ROOT}" \
    CACHE_DIR="${PREPARED_ROOT}/cache" \
    SKY_MASK_ROOT="${PREPARED_ROOT}/sky_masks" \
    COLMAP_DIR="${colmap_dir}" \
    COLMAP_POINTS3D="${colmap_dir}/triangulated_text/points3D.txt" \
    CAS_TRACK_PATH="${castrack_path}" \
    ACTOR_BOX_SCALE=1.0 \
    SPLIT_TYPE="${SPLIT_TYPE}" \
    TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION}" \
    LIDAR_INITIALIZATION_FRAMES="${LIDAR_INITIALIZATION_FRAMES}" \
    OUTPUT_DIR="${output_dir}" \
    CONFIG="${CONFIG}" \
    ARMGS_PYTHON="${ARMGS_PYTHON}" \
    GPU_ID="${gpu}" \
    PAPER_MODE=0 \
    ITERATIONS="${ITERATIONS}" \
    RESUME="${resume}" \
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
    LOG_INTERVAL="${LOG_INTERVAL}" \
    IMAGE_LOG_INTERVAL="${IMAGE_LOG_INTERVAL}" \
    EVAL_INTERVAL=0 \
    WANDB_ENABLED="${WANDB_ENABLED}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_RUN_NAME="${run_name}" \
    WANDB_MODE="${WANDB_MODE}" \
    WANDB_DIR="${WANDB_DIR}" \
    "${TRAIN_LAUNCHER}" "${sequence}" "${start}" "${end}" \
    "${extra_args[@]}"
}

run_scene_body() {
  local gpu="$1"
  local entry="$2"
  local sequence start end output_dir final_checkpoint resume eval_only=0
  IFS=':' read -r sequence start end <<< "${entry}"
  output_dir="${OUTPUT_ROOT}/${sequence}/${OUTPUT_TAG}"
  final_checkpoint="${output_dir}/checkpoints/$(final_checkpoint_name)"

  if scene_complete "${output_dir}" && [[ "${SKIP_COMPLETED}" == "1" ]]; then
    printf '[GPU %s][%s] complete; skipping\n' "${gpu}" "${sequence}"
    return 0
  fi

  resume="$(latest_checkpoint "${output_dir}")"
  if [[ -n "${resume}" && "${AUTO_RESUME}" != "1" ]]; then
    printf 'error: [%s] checkpoint exists but AUTO_RESUME=0: %s\n' \
      "${sequence}" "${resume}" >&2
    return 1
  fi
  if [[ -s "${final_checkpoint}" ]]; then
    resume="${final_checkpoint}"
    eval_only=1
    printf '[GPU %s][%s] final checkpoint found; running missing/re-requested evaluation only\n' \
      "${gpu}" "${sequence}"
  elif [[ -n "${resume}" ]]; then
    printf '[GPU %s][%s] resuming %s\n' "${gpu}" "${sequence}" "${resume}"
  elif [[ -s "${output_dir}/run_metadata.json" || \
          -s "${output_dir}/resolved_config.yaml" || \
          -s "${output_dir}/wandb_run.json" ]]; then
    printf 'error: [%s] partial run has no checkpoint; refusing to overwrite %s\n' \
      "${sequence}" "${output_dir}" >&2
    return 1
  else
    printf '[GPU %s][%s] starting a fresh run\n' "${gpu}" "${sequence}"
  fi

  wait_for_gpu "${gpu}"
  prepare_scene "${gpu}" "${sequence}" "${start}" "${end}" "${output_dir}"
  wait_for_gpu "${gpu}"
  train_scene \
    "${gpu}" "${sequence}" "${start}" "${end}" "${output_dir}" \
    "${resume}" "${eval_only}"

  if ! scene_complete "${output_dir}"; then
    printf 'error: [%s] launcher exited without final checkpoint and both evaluations\n' \
      "${sequence}" >&2
    return 1
  fi
  printf '[GPU %s][%s] training and final evaluation complete\n' \
    "${gpu}" "${sequence}"
}

run_scene_logged() {
  local gpu="$1"
  local entry="$2"
  local sequence _ _ log_dir log_path status
  local -a pipeline_status=()
  IFS=':' read -r sequence _ _ <<< "${entry}"
  log_dir="${LOG_ROOT}/${sequence}"
  log_path="${log_dir}/launcher.log"
  mkdir -p -- "${log_dir}"

  set +e
  (
    set -Eeuo pipefail
    run_scene_body "${gpu}" "${entry}"
  ) 2>&1 | tee -a "${log_path}"
  pipeline_status=("${PIPESTATUS[@]}")
  status="${pipeline_status[0]}"
  if [[ "${status}" == "0" && "${pipeline_status[1]}" != "0" ]]; then
    status="${pipeline_status[1]}"
  fi
  set -e
  return "${status}"
}

run_queue() {
  local gpu="$1"
  shift
  local entry sequence _ _
  local -a failures=()
  for entry in "$@"; do
    IFS=':' read -r sequence _ _ <<< "${entry}"
    printf '[GPU %s] dequeued %s\n' "${gpu}" "${sequence}"
    if run_scene_logged "${gpu}" "${entry}"; then
      printf '[GPU %s] finished %s\n' "${gpu}" "${sequence}"
    else
      failures+=("${sequence}")
      printf '[GPU %s] FAILED %s\n' "${gpu}" "${sequence}" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        break
      fi
    fi
  done
  if (( ${#failures[@]} > 0 )); then
    printf '[GPU %s] failed scene(s): %s\n' "${gpu}" "${failures[*]}" >&2
    return 1
  fi
  printf '[GPU %s] queue complete\n' "${gpu}"
}

if [[ "${RUN_PREPARE}" != "1" ]]; then
  for entry in "${GPU0_QUEUE[@]}" "${GPU1_QUEUE[@]}"; do
    IFS=':' read -r sequence start end <<< "${entry}"
    scene_assets_ready "${sequence}" "${start}" "${end}" ||
      die "prepared sky/COLMAP assets are incomplete for ${sequence}"
  done
fi

WORKER_PIDS=()
terminate_workers() {
  trap - INT TERM
  printf 'Interrupt received; terminating GPU workers...\n' >&2
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}
trap terminate_workers INT TERM

run_queue "${GPU_ID_VALUES[0]}" "${GPU0_QUEUE[@]}" &
WORKER_PIDS+=("$!")
run_queue "${GPU_ID_VALUES[1]}" "${GPU1_QUEUE[@]}" &
WORKER_PIDS+=("$!")

batch_status=0
for worker_pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${worker_pid}"; then
    batch_status=1
  fi
done
trap - INT TERM

if (( batch_status != 0 )); then
  printf 'ArmGS Waymo batch finished with one or more failed scenes.\n' >&2
  exit "${batch_status}"
fi
printf 'ArmGS Waymo batch complete: all selected scenes finished.\n'
