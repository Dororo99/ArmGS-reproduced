# Grounded SAM sky mask 설정

ArmGS는 렌더링된 Gaussian opacity의 여집합을 sky 확률로 사용하고, 미리 추출한
sky mask와 BCE를 계산한다. 논문 재현 경로는 Street Gaussians와 동일하게
GroundingDINO의 `sky` 박스를 SAM box prompt로 전달한다. actor pixel mask는 이
과정에 필요하지 않으며, foreground 항은 렌더링된 actor alpha의 entropy이다.

- [ArmGS 방법론](https://arxiv.org/html/2507.03886#S3)
- [Street Gaussians sky/foreground loss](https://arxiv.org/html/2401.01339#A1.SS1)
- [공식 Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## 1. 독립 Conda 환경과 모델 준비

학습용 `camosplat` 환경을 변경하지 않는다. 아래 스크립트가 `armgs-gsam`이라는
독립 Conda 환경을 만들고 원본 Grounded SAM commit, PyTorch CUDA 11.8 패키지,
GroundingDINO CUDA extension과 세 모델 자산을 설치한다.

```bash
cd /workspace/projects/camosplat/dohyun/ArmGS
bash scripts/setup_grounded_sam.sh
```

직접 환경 안에서 명령을 실행하려면 활성화한다. Grounded SAM setup/generation
wrapper는 `conda run`을 사용하므로 활성화하지 않아도 된다.

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate armgs-gsam
```

이 서버의 Conda 설정에서는 실제 환경 prefix가 `/venv/armgs-gsam`이 된다. 설치
결과는 다음 위치에 있고 Git에는 포함되지 않는다.

```text
third_party/Grounded-Segment-Anything/       # 고정된 원본 코드
checkpoints/grounded_sam/
├── groundingdino_swint_ogc.pth
├── sam_vit_h_4b8939.pth
├── bert-base-uncased/
└── huggingface/                             # 다운로드 cache
```

스크립트는 다시 실행해도 기존 환경·checkout·완료된 가중치를 재사용한다. 중단된
checkpoint는 `<파일>.part`에서 `aria2c`(없으면 `curl -C -`)로 이어받고, 완료 후 SHA-256 sidecar를
기록한다. 코드만 먼저 설치하려면 다음을 사용한다.

```bash
SKIP_MODEL_DOWNLOADS=1 bash scripts/setup_grounded_sam.sh
```

기본 CUDA/GCC 경로는 `/usr/local/cuda-11.8`, `/usr/bin/gcc-11`,
`/usr/bin/g++-11`이다. 다른 서버에서는 `CUDA_HOME`, `GSAM_CC`, `GSAM_CXX`로
같은 버전의 경로를 지정한다.

## 2. scene-0061 입력 확인

모델을 읽지 않고 39 capture × 6 camera, 총 234개 입력과 출력 경로만 확인할 수
있다.

```bash
DRY_RUN=1 bash scripts/generate_nuscenes_scene_0061_sky_masks.sh 0
```

기본 데이터셋은 `data/nuscenes`, metadata version은 `v1.0-trainval`이다. 다른
위치라면 다음처럼 지정한다.

```bash
NUSCENES_ROOT=/absolute/path/to/nuscenes \
  DRY_RUN=1 bash scripts/generate_nuscenes_scene_0061_sky_masks.sh 0
```

## 3. mask 생성

GPU 0에서 전체 생성과 QA 자료 생성을 시작한다.

```bash
bash scripts/generate_nuscenes_scene_0061_sky_masks.sh 0
```

scene-0061 wrapper의 기본값은 positive prompt `sky`, exclusion prompt
`building . tree`, box threshold `0.3`, text threshold `0.25`, SAM ViT-H이다.
건물과 수목으로 검출한 SAM 영역은 sky union에서 제거해 건축물·canopy의 sky
오검출을 줄인다. 범용 Python CLI의 `--negative-text-prompt`는 기본적으로
비활성화되어 기존 동작과 호환된다.
wrapper에서도 exclusion을 끄려면 빈 환경 변수를 명시한다.

```bash
NEGATIVE_TEXT_PROMPT= bash scripts/generate_nuscenes_scene_0061_sky_masks.sh 0
```

유효한 기존 mask는 검사한 뒤 건너뛰므로 중단 후 같은 명령으로 resume할 수 있다.
threshold나 exclusion prompt를 바꾼 기존 결과에는 `OVERWRITE=1`을 사용한다.

```bash
NEGATIVE_TEXT_PROMPT='building . tree' OVERWRITE=1 bash scripts/generate_nuscenes_scene_0061_sky_masks.sh 0
```

최종 training mask 경로는 다음과 같다.

```text
data/sky_masks/nuscenes/v1.0-trainval/scene-0061/
├── CAM_FRONT/<sample_data_token>.png
├── CAM_FRONT_LEFT/<sample_data_token>.png
├── CAM_FRONT_RIGHT/<sample_data_token>.png
├── CAM_BACK/<sample_data_token>.png
├── CAM_BACK_LEFT/<sample_data_token>.png
└── CAM_BACK_RIGHT/<sample_data_token>.png
```

각 파일은 원본과 동일한 `900×1600` single-channel PNG이며 `255=sky`,
`0=non-sky`이다. sky detection의 SAM mask를 OR로 합친 뒤 optional exclusion
mask union을 빼서 `sky & ~exclusion`을 저장한다. sky detection이 없으면 all-zero
mask를 저장하며, resize된 overlay는 학습 입력으로 사용하지 않는다.

## 4. QA와 학습 연결

생성 뒤 두 파일을 먼저 확인한다.

```text
.../scene-0061/generation_manifest.json
.../scene-0061/sky_mask_contact_sheet.jpg
```
manifest schema 2에는 generation별 positive/exclusion prompt와 frame별 sky
`detection_count`/`phrases`/`logits`, exclusion
`excluded_detection_count`/`excluded_phrases`/`excluded_logits`, 최종 sky
coverage, no-detection 여부와 에러가 기록된다. resume 시 기존 frame의 exclusion
metadata도 승계된다. contact sheet에서는 건물·수목 경계, 태양/구름, 후방
카메라와 all-zero/all-sky에 가까운 mask를 우선 검수한다.
prompt나 threshold를 조정한 뒤에는 OVERWRITE=1로 다시 생성한다.

학습 launcher는 위 scene directory를 `SKY_MASK_ROOT` 기본값으로 사용한다.
sky mask가 234개 모두 생성된 뒤 기존 명령을 실행하면 된다. 시각 QA에서 명백한
오검출로 확인한 18개 sample-data token은
`configs/nuscenes_scene_0061_sky_mask_reject_tokens.txt`가 기본 적용한다. PNG를
all-zero로 바꾸지 않고 그대로 로드하되 해당 프레임의 sky BCE만 0으로 만든다.
따라서 strict auxiliary 검증과 원본 mask provenance는 유지된다.

```bash
bash scripts/train_nuscenes_scene_0061.sh 0
```

W&B에는 `LOG_INTERVAL=100`마다 100-step training 구간 평균 scalar를 기록하고,
`IMAGE_LOG_INTERVAL=500`마다 별도로 `train/gt_vs_render` 비교 이미지를 기록한다.
이미지 주기는 `IMAGE_LOG_INTERVAL=0`으로 끌 수 있다. 이때 `train/ssim_loss`는
학습 objective의 `1 - SSIM`이며 held-out SSIM 평가값이 아니다. 별도 held-out
평가는 capture-safe eval 30장을 novel-view
embedding(`training_row=None`)으로 렌더링해 PSNR, SSIM, projected actor-box PSNR,
LPIPS를 aggregate/per-camera로 계산한다. 기본값은 `EVAL_INTERVAL=1000`,
`EVAL_AT_END=1`, `EVAL_LPIPS=1`이며 W&B `eval/*`와 다음 파일에 함께 저장된다.

```text
<OUTPUT_DIR>/evaluation/step_XXXXXXXX.json
<OUTPUT_DIR>/evaluation/step_XXXXXXXX_<CAMERA>_gt_render.png
```

기존 30K checkpoint만 평가하려면 원래 `training.log`를 건드리지 않는 eval-only
경로를 사용한다. 결과 로그는 `evaluation.log`에 append된다.

```bash
EVAL_ONLY=1 RESUME=/path/to/final.pt OUTPUT_DIR=/path/to/run \
  bash scripts/train_nuscenes_scene_0061.sh 0
```

LPIPS를 끄려면 `EVAL_LPIPS=0`, periodic 평가만 끄고 final 평가는 유지하려면
`EVAL_INTERVAL=0`을 사용한다.

W&B failure/resume와 final checkpoint Artifact 정책을 포함한 전체 로깅 계약은
[`docs/WANDB_LOGGING.ko.md`](WANDB_LOGGING.ko.md)를 참고한다.

다른 mask root를 쓰는 경우 directory는 camera channel 바로 위여야 한다.

```bash
SKY_MASK_ROOT=/absolute/path/to/scene-0061 \
  bash scripts/train_nuscenes_scene_0061.sh 0
```

다른 검수 목록으로 교체하거나 reject를 완전히 끄려면 각각 다음처럼 실행한다.

```bash
SKY_MASK_REJECT_LIST=/absolute/path/to/reject_tokens.txt \
  bash scripts/train_nuscenes_scene_0061.sh 0

SKY_MASK_REJECT_LIST= bash scripts/train_nuscenes_scene_0061.sh 0
```

목록은 UTF-8 text이며 빈 줄, `#` comment, 줄 끝 comment를 허용한다. 각 token은
중복 없는 32자리 hex여야 하고 선택한 scene/camera에 없는 token은 학습 시작 전에
오류로 처리된다. reject 파일 경로·내용 digest·적용 개수는 checkpoint dataset
identity, `run_metadata.json`, W&B dataset config에 기록된다.

scene-0061 설정은 strict auxiliary supervision, `lambda_depth=0.01`,
`lambda_sky=0.05`, `lambda_foreground=0.1`을 사용한다. 여기서 strict auxiliary는
LiDAR depth와 sky mask를 요구하지만 actor pixel mask를 요구하지 않는다.

## 문제 해결

- 다운로드가 끊기면 setup 스크립트를 그대로 재실행한다. `.part` 파일을 수동으로
  지울 필요가 없다.
- `groundingdino._C` import 오류가 나면 CUDA 11.8과 GCC/G++ 11 경로를 확인한 뒤
  setup 스크립트를 다시 실행한다.
- 생성기는 mask 하나를 원자적으로 저장하고 manifest를 매 frame 갱신한다. 실패
  원인은 manifest의 마지막 `error` record에서도 확인할 수 있다.
- threshold를 변경한 결과와 기존 mask를 섞지 않으려면 `OVERWRITE=1`로 scene 전체를
  다시 생성한다.
