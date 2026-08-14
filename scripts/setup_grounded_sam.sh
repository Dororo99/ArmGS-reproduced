#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_grounded_sam.sh

Create the isolated armgs-gsam Conda environment, install the pinned original
Grounded SAM implementation, and download all offline inference assets.

Optional environment overrides:
  CONDA_BIN        mamba/conda executable (auto-detected)
  ENV_NAME         Conda environment name (default: armgs-gsam)
  GSAM_ROOT        Grounded SAM checkout directory
  CHECKPOINT_DIR   Model/checkpoint directory
  CUDA_HOME        CUDA toolkit root (default: /usr/local/cuda-11.8)
  GSAM_CC          C compiler (default: /usr/bin/gcc-11)
  GSAM_CXX         C++ compiler (default: /usr/bin/g++-11)
  SKIP_MODEL_DOWNLOADS=1
                   Install code but skip GroundingDINO/SAM/BERT downloads

Downloads are resumable. Existing completed assets and an environment that is
already usable are reused, so the script can safely be run again.
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
if (( $# != 0 )); then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ARMGS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

ENV_NAME="${ENV_NAME:-armgs-gsam}"
GSAM_ROOT="${GSAM_ROOT:-${ARMGS_ROOT}/third_party/Grounded-Segment-Anything}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ARMGS_ROOT}/checkpoints/grounded_sam}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
GSAM_CC="${GSAM_CC:-/usr/bin/gcc-11}"
GSAM_CXX="${GSAM_CXX:-/usr/bin/g++-11}"
SKIP_MODEL_DOWNLOADS="${SKIP_MODEL_DOWNLOADS:-0}"

if [[ "${GSAM_ROOT}" != /* ]]; then
  GSAM_ROOT="${ARMGS_ROOT}/${GSAM_ROOT}"
fi
if [[ "${CHECKPOINT_DIR}" != /* ]]; then
  CHECKPOINT_DIR="${ARMGS_ROOT}/${CHECKPOINT_DIR}"
fi

GSAM_REPOSITORY="https://github.com/IDEA-Research/Grounded-Segment-Anything.git"
GSAM_COMMIT="126abe633ffe333e16e4a0a4e946bc1003caf757"
GROUNDINGDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
# The current Hugging Face CDN intermittently closes TLS connections on this
# host. These are Hugging Face's canonical legacy BERT artifacts and contain
# everything AutoTokenizer/BertModel need for fully offline local loading.
BERT_CONFIG_URL="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
BERT_WEIGHTS_URL="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin"
BERT_VOCAB_URL="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"

GROUNDINGDINO_CHECKPOINT="${CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT="${CHECKPOINT_DIR}/sam_vit_h_4b8939.pth"
BERT_DIR="${CHECKPOINT_DIR}/bert-base-uncased"
GSAM_HF_HOME="${CHECKPOINT_DIR}/huggingface"

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

CONDA_BIN="$(resolve_conda_bin)"

[[ "${ENV_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "ENV_NAME contains unsupported characters: ${ENV_NAME}"
[[ "${SKIP_MODEL_DOWNLOADS}" == "0" || "${SKIP_MODEL_DOWNLOADS}" == "1" ]] || \
  die "SKIP_MODEL_DOWNLOADS must be 0 or 1"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || \
  die "CUDA nvcc was not found at ${CUDA_HOME}/bin/nvcc"
"${CUDA_HOME}/bin/nvcc" --version | grep -Eq 'release 11\.8([, ]|$)' || \
  die "CUDA 11.8 is required; override CUDA_HOME only with a CUDA 11.8 toolkit"
[[ -x "${GSAM_CC}" ]] || die "GCC 11 was not found: ${GSAM_CC}"
[[ -x "${GSAM_CXX}" ]] || die "G++ 11 was not found: ${GSAM_CXX}"
[[ "$("${GSAM_CC}" -dumpversion)" == 11* ]] || \
  die "GSAM_CC must point to GCC 11: ${GSAM_CC}"
[[ "$("${GSAM_CXX}" -dumpversion)" == 11* ]] || \
  die "GSAM_CXX must point to G++ 11: ${GSAM_CXX}"

mkdir -p -- "$(dirname -- "${GSAM_ROOT}")" "${CHECKPOINT_DIR}" "${GSAM_HF_HOME}"

run_in_env() {
  env \
    CUDA_HOME="${CUDA_HOME}" \
    CC="${GSAM_CC}" \
    CXX="${GSAM_CXX}" \
    TORCH_CUDA_ARCH_LIST="8.9" \
    AM_I_DOCKER=False \
    BUILD_WITH_CUDA=True \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME="${GSAM_HF_HOME}" \
    HF_HUB_CACHE="${GSAM_HF_HOME}/hub" \
    TRANSFORMERS_CACHE="${GSAM_HF_HOME}/transformers" \
    GSAM_BERT_DIR="${BERT_DIR}" \
    "${CONDA_BIN}" run -n "${ENV_NAME}" "$@"
}

if run_in_env python -c 'import sys; assert sys.version_info[:2] == (3, 10)' \
    >/dev/null 2>&1; then
  printf 'Reusing Conda environment: %s\n' "${ENV_NAME}"
elif "${CONDA_BIN}" run -n "${ENV_NAME}" true >/dev/null 2>&1; then
  printf 'Updating Conda environment to Python 3.10: %s\n' "${ENV_NAME}"
  "${CONDA_BIN}" install --yes --override-channels --channel conda-forge \
    --name "${ENV_NAME}" \
    python=3.10 pip setuptools wheel ninja
else
  printf 'Creating Conda environment: %s\n' "${ENV_NAME}"
  "${CONDA_BIN}" create --yes --override-channels --channel conda-forge \
    --name "${ENV_NAME}" \
    python=3.10 pip setuptools wheel ninja
fi

printf 'Installing pinned Python and PyTorch dependencies...\n'
run_in_env python -m pip install --upgrade \
  'pip==23.3.2' \
  'setuptools==68.2.2' \
  'wheel==0.41.3' \
  'ninja==1.11.1.1'

run_in_env python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.0.1' \
  'torchvision==0.15.2'

run_in_env python -m pip install \
  'numpy==1.26.4' \
  'transformers==4.33.2' \
  'huggingface-hub==0.17.3' \
  'tokenizers==0.13.3' \
  'timm==0.9.7' \
  'opencv-python-headless==4.8.1.78' \
  'opencv-python==4.8.1.78'  \
  'pycocotools==2.0.7' \
  'matplotlib==3.8.0' \
  'scipy==1.11.3' \
  'addict==2.4.0' \
  'yapf==0.40.1' \
  'supervision==0.22.0'

if [[ ! -e "${GSAM_ROOT}" ]]; then
  printf 'Cloning Grounded SAM at pinned commit %s...\n' "${GSAM_COMMIT}"
  git clone --filter=blob:none "${GSAM_REPOSITORY}" "${GSAM_ROOT}"
elif [[ ! -d "${GSAM_ROOT}/.git" ]]; then
  die "GSAM_ROOT exists but is not a Git checkout: ${GSAM_ROOT}"
fi

if ! git -C "${GSAM_ROOT}" cat-file -e "${GSAM_COMMIT}^{commit}" 2>/dev/null; then
  printf 'Fetching pinned Grounded SAM commit...\n'
  git -C "${GSAM_ROOT}" fetch --depth 1 origin "${GSAM_COMMIT}"
fi

CURRENT_COMMIT="$(git -C "${GSAM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${CURRENT_COMMIT}" != "${GSAM_COMMIT}" ]]; then
  if ! git -C "${GSAM_ROOT}" diff --quiet || \
      ! git -C "${GSAM_ROOT}" diff --cached --quiet; then
    die "Grounded SAM has tracked local changes; preserve them before checkout"
  fi
  git -C "${GSAM_ROOT}" checkout --detach "${GSAM_COMMIT}"
fi

[[ -f "${GSAM_ROOT}/segment_anything/setup.py" ]] || \
  die "segment_anything checkout is incomplete: ${GSAM_ROOT}"
[[ -f "${GSAM_ROOT}/GroundingDINO/setup.py" ]] || \
  die "GroundingDINO checkout is incomplete: ${GSAM_ROOT}"

printf 'Installing SAM and the GroundingDINO CUDA extension...\n'
run_in_env python -m pip install -e "${GSAM_ROOT}/segment_anything"
run_in_env python -m pip install --no-build-isolation --no-deps \
  -e "${GSAM_ROOT}/GroundingDINO"

# Guard against an unconstrained transitive dependency upgrading NumPy to 2.x.
run_in_env python -m pip install 'numpy==1.26.4'
run_in_env python -m pip check

download_resumable() {
  local url="$1"
  local destination="$2"
  local minimum_bytes="$3"
  local partial="${destination}.part"
  local size

  if [[ -s "${destination}" ]]; then
    size="$(stat -c '%s' "${destination}")"
    (( size >= minimum_bytes )) || \
      die "existing checkpoint is unexpectedly small: ${destination} (${size} bytes)"
    if [[ -f "${destination}.sha256" ]]; then
      (cd -- "$(dirname -- "${destination}")" && \
        sha256sum --check --status "$(basename -- "${destination}").sha256") || \
        die "checkpoint SHA-256 verification failed: ${destination}"
    else
      (cd -- "$(dirname -- "${destination}")" && \
        sha256sum "$(basename -- "${destination}")" > \
          "$(basename -- "${destination}").sha256")
    fi
    printf 'Reusing checkpoint: %s\n' "${destination}"
    return
  fi

  printf 'Downloading (or resuming): %s\n' "${destination}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --max-connection-per-server=16 --split=16 \
      --min-split-size=16M --file-allocation=none \
      --dir="$(dirname -- "${partial}")" --out="$(basename -- "${partial}")" \
      "${url}"
  else
    command -v curl >/dev/null 2>&1 || \
      die "aria2c or curl is required for checkpoint downloads"
    curl --fail --location --retry 5 --retry-delay 2 --retry-all-errors \
      --continue-at - --output "${partial}" "${url}"
  fi
  size="$(stat -c '%s' "${partial}")"
  (( size >= minimum_bytes )) || \
    die "downloaded checkpoint is unexpectedly small: ${partial} (${size} bytes)"
  mv -- "${partial}" "${destination}"
  (cd -- "$(dirname -- "${destination}")" && \
    sha256sum "$(basename -- "${destination}")" > \
      "$(basename -- "${destination}").sha256")
}

if [[ "${SKIP_MODEL_DOWNLOADS}" == "0" ]]; then
  download_resumable "${GROUNDINGDINO_URL}" "${GROUNDINGDINO_CHECKPOINT}" 500000000
  download_resumable "${SAM_URL}" "${SAM_CHECKPOINT}" 2000000000

  printf 'Downloading (or reusing) bert-base-uncased: %s\n' "${BERT_DIR}"
  mkdir -p -- "${BERT_DIR}"
  download_resumable "${BERT_CONFIG_URL}" \
    "${BERT_DIR}/config.json" 400
  download_resumable "${BERT_WEIGHTS_URL}" \
    "${BERT_DIR}/pytorch_model.bin" 400000000
  download_resumable "${BERT_VOCAB_URL}" \
    "${BERT_DIR}/vocab.txt" 200000
else
  printf 'Skipping model downloads because SKIP_MODEL_DOWNLOADS=1\n'
fi

printf 'Verifying the environment and compiled GroundingDINO extension...\n'
run_in_env python -c \
  'import torch; import groundingdino; import segment_anything; from groundingdino import _C; assert torch.__version__.startswith("2.0.1"); assert torch.version.cuda == "11.8"; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available=True")'
if [[ -f "${BERT_DIR}/config.json" && \
      -f "${BERT_DIR}/pytorch_model.bin" && \
      -f "${BERT_DIR}/vocab.txt" ]]; then
  run_in_env python -c \
    'import sys; from transformers import AutoTokenizer, BertModel; path = sys.argv[1]; tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True); model = BertModel.from_pretrained(path, local_files_only=True); assert tokenizer.vocab_size == 30522; assert model.config.hidden_size == 768; print(f"offline_bert={path} vocab={tokenizer.vocab_size} hidden={model.config.hidden_size}")' \
    "${BERT_DIR}"
elif [[ "${SKIP_MODEL_DOWNLOADS}" == "0" ]]; then
  die "offline BERT assets are incomplete: ${BERT_DIR}"
else
  printf 'Skipping offline BERT verification because model downloads were skipped.\n'
fi
ENV_PREFIX="$(run_in_env python -c 'import sys; print(sys.prefix)')"

printf '\nGrounded SAM setup complete.\n'
printf '  Conda environment: %s\n' "${ENV_NAME}"
printf '  Conda prefix: %s\n' "${ENV_PREFIX}"
printf '  Grounded SAM: %s @ %s\n' "${GSAM_ROOT}" "${GSAM_COMMIT}"
printf '  Checkpoints: %s\n' "${CHECKPOINT_DIR}"
printf '  BERT: %s\n' "${BERT_DIR}"
printf 'Next: scripts/generate_nuscenes_scene_0061_sky_masks.sh 0\n'
