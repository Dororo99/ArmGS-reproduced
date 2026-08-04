# W&B 로깅 계약

nuScenes scene-0061 launcher는 기본적으로 `CamoSplat/ArmGS-nuScenes`에
기록한다. training scalar, training preview, held-out 평가는 서로 다른 주기를
사용하며 한 옵션의 주기를 바꿔도 나머지 주기는 바뀌지 않는다.

| 작업 | launcher 변수 | 기본값 | 동작 |
|---|---|---:|---|
| training scalar | `LOG_INTERVAL` | 100 | 직전 구간의 training 통계를 한 번 기록 |
| training preview | `IMAGE_LOG_INTERVAL` | 500 | 현재 training batch의 `GT | render` 한 장 기록; `0`은 비활성화 |
| held-out 평가 | `EVAL_INTERVAL` | 1,000 | eval split 전체 metric 및 카메라별 preview 기록; `0`은 periodic만 비활성화 |
| local checkpoint | `CHECKPOINT_INTERVAL` | 1,000 | W&B와 무관하게 로컬 `.pt` 저장 |

`EVAL_AT_END=1`이 기본이므로 `EVAL_INTERVAL=0`이어도 학습 종료 시 held-out
평가는 한 번 실행한다. 반대로 training preview는 final step이 500의 배수가
아니면 마지막에 강제로 추가하지 않는다.

## 기록 항목

training objective는 `train/*` namespace에 기록한다. `train/loss`는 Eq. (9)의
가중 합이고, `train/rgb_l1`, `train/ssim_loss`, `train/depth_loss`,
`train/sky_loss`, `train/foreground_loss`는 각 가중치를 곱하기 전 loss다.
특히 `train/ssim_loss`는 `1 - SSIM`이므로 값이 작을수록 좋으며 평가용 SSIM과
혼동하면 안 된다.

`LOG_INTERVAL=100`이면 위 loss는 100번째 이미지 한 장의 순간값이 아니라 직전
100번 optimizer step의 산술평균이다. 첫 기록이나 resume 직후처럼 구간이
짧으면 실제 누적 step 수만 평균하며, final step의 불완전한 구간도 한 번 flush
한다. 함께 기록하는 진단 key는 다음과 같다.

- `train/gaussians/total`, `train/gaussians/background`,
  `train/gaussians/actors`, `train/gaussians/actor/<ID>`: 기록 시점의 Gaussian 수
- `train/lr/<OPTIMIZER_GROUP>`: 기록 시점의 optimizer group별 learning rate
- `train/telemetry/window_steps`: 해당 평균과 timing에 포함된 새 optimizer step 수
- `train/performance/step_time_seconds`, `train/performance/steps_per_second`:
  직전 기록 구간의 처리 시간과 throughput
- `train/cuda/memory_allocated_mib`, `train/cuda/memory_reserved_mib`: 기록 시점의
  CUDA memory
- `train/cuda/max_memory_allocated_mib`,
  `train/cuda/max_memory_reserved_mib`: 해당 process에서 관찰된 lifetime peak
- `train/density/steps_with_results`, `updated_groups`,
  `topology_changed_groups`, `duplicated_gaussians`, `split_parent_gaussians`,
  `split_child_gaussians`, `pruned_gaussians`, `opacity_reset_groups`: 직전 구간의
  density-control 합계. 해당 구간에 density 결과가 하나도 없으면 생략

`train/gt_vs_render`는 `IMAGE_LOG_INTERVAL`의 배수 step에서 사용한 현재 training
batch 한 장을 좌측 GT, 우측 render로 붙인 이미지다. metric 계산용 eval image가
아니며, 별도의 추가 training step을 수행하지 않는다. 같은 record에는
`train/image_camera_id`, `train/image_camera`, `train/image_timestamp_ns`와 가능한
경우 `train/image_training_row`, `train/image_frame_index`를 함께 기록해 어떤
입력인지 추적할 수 있다.

held-out 평가는 다음 key를 기록한다.

- `eval/psnr`, `eval/ssim`, `eval/lpips`, `eval/actor_psnr`
- `eval/image_count`, `eval/pixel_count`, `eval/actor_image_count`,
  `eval/actor_pixel_count`
- 위 항목의 `eval/<CAMERA>/...` 카메라별 값
- 카메라별 최초 평가 frame의 `eval/<CAMERA>/gt_vs_render`

aggregate metric은 eval 이미지 전체의 pixel/image count를 사용해 카메라별 값을
가중 결합한다. 같은 결과와 preview PNG는
`<OUTPUT_DIR>/evaluation/step_XXXXXXXX.*`에도 저장되므로 W&B가 끊겨도 로컬
평가 결과는 남는다.

## 오류 정책

`WANDB_FAIL_FAST=0`이 기본이다. W&B init/log/image/summary/Artifact/finish가
실패하면 경고를 남기고 학습, 로컬 평가, 로컬 checkpoint를 계속한다. 실험 추적
실패도 학습 실패로 취급해야 하는 환경에서만
`WANDB_FAIL_FAST=1`을 사용한다.

~~~bash
WANDB_FAIL_FAST=1 bash scripts/train_nuscenes_scene_0061.sh 0
~~~

`WANDB_MODE=offline`은 `<WANDB_DIR>`에 run을 쌓고 나중에 sync할 때 사용한다.
`WANDB_MODE=disabled`는 W&B 기록을 비활성화한다.

## resume와 run ID

checkpoint의 exact resume와 W&B history resume는 서로 별개지만, 동일한
`OUTPUT_DIR`에서는 둘을 수동으로 맞출 필요가 없다. trainer는 다음 우선순위로
W&B run ID를 결정한다.

1. `WANDB_RUN_ID` 또는 `--wandb-run-id`로 전달한 explicit ID
2. `<OUTPUT_DIR>/wandb_run.json`에 저장된 기존 run ID
3. 둘 다 없으면 새 run

1번은 2번을 명시적으로 덮어쓴다. 1번이나 2번에서 ID를 찾으면
`wandb.init(id=..., resume="allow")`로 이어 쓰고, 새 run도 초기화 직후 sidecar를
만든다. sidecar에는 실제 run ID, entity, project, name, URL, mode뿐 아니라
`resume_source=explicit|sidecar|new`와 resume 요청 여부를 기록한다.

~~~bash
OUTPUT_DIR=/path/to/run WANDB_RUN_NAME=scene0061-main \
  bash scripts/train_nuscenes_scene_0061.sh 0

OUTPUT_DIR=/path/to/run RESUME=/path/to/run/checkpoints/step_00010000.pt \
  WANDB_RUN_NAME=scene0061-main \
  bash scripts/train_nuscenes_scene_0061.sh 0
~~~

두 번째 실행은 같은 `OUTPUT_DIR`의 sidecar ID를 자동으로 사용한다. 기존 30K
실험처럼 이 기능 추가 전에 만들어져 sidecar가 없는 run은 display name이나
checkpoint만으로 W&B ID를 안전하게 추론할 수 없다. 해당 run ID를 알고 있다면
다음처럼 explicit override를 한 번 전달해야 한다.

~~~bash
OUTPUT_DIR=/path/to/legacy-run RESUME=/path/to/legacy-run/checkpoints/final.pt \
  WANDB_RUN_ID=existing-wandb-run-id \
  bash scripts/train_nuscenes_scene_0061.sh 0
~~~

trainer step을 W&B global step으로 그대로 사용한다. 복구된 checkpoint step 자체의
training callback은 다시 실행하지 않고 그 다음 새 optimizer step부터 history를
추가한다. 서로 다른 실험의 sidecar나 explicit ID를 재사용하지 않아야 한다.

## checkpoint Artifact

`WANDB_LOG_CHECKPOINT=0`이 기본이다. 이때 checkpoint는 로컬에만 남고 W&B
summary에는 final checkpoint 경로와 완료 step 같은 metadata만 기록한다.

`WANDB_LOG_CHECKPOINT=1`은 periodic checkpoint를 모두 업로드하지 않는다. 정상
학습을 마친 뒤 `<OUTPUT_DIR>/checkpoints/final.pt` 하나만 model Artifact로
기록하며 SHA-256, file size, step, scene과 dataset identity를 metadata에 넣는다.
`EVAL_ONLY=1`에서는 checkpoint Artifact를 새로 올리지 않는다. 큰 checkpoint의
저장 시간·대역폭·Artifact quota를 명시적으로 감수할 때만 활성화한다.
성공한 upload의 checksum과 크기는 W&B summary의
`checkpoint_artifact/sha256`, `checkpoint_artifact/size_bytes`에도 남는다.

~~~bash
WANDB_LOG_CHECKPOINT=1 bash scripts/train_nuscenes_scene_0061.sh 0
~~~

## 기본 권장값

일반 학습은 다음과 동일하다.

~~~bash
LOG_INTERVAL=100 IMAGE_LOG_INTERVAL=500 EVAL_INTERVAL=1000 \
  WANDB_FAIL_FAST=0 WANDB_LOG_CHECKPOINT=0 \
  bash scripts/train_nuscenes_scene_0061.sh 0
~~~

이미지 업로드 비용을 줄이려면 `IMAGE_LOG_INTERVAL`을 늘리거나 `0`으로 설정한다.
scalar를 더 자주 기록하는 것은 이미지 렌더링·업로드 주기에 영향을 주지 않는다.
