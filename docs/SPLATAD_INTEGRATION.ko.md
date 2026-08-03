# SplatAD 통합 설계 메모

이 문서는 독립 ArmGS vertical slice를 인접 SplatAD/CamoSplat 코드베이스에 연결할 때
필요한 결정과 삽입 지점을 기록한다. 독립 패키지 안에서는 composite scene, cubemap
sky, gsplat CUDA 전체 backward와 one-view trainer가 동작한다. 다만 실제
SplatAD/Nerfstudio model 및 dataparser에는 아직 삽입하지 않았다.

## 통합 전에 고정할 세 가지 결정

### 1. RGB SH 경로를 별도로 만든다

ArmGS는 SH를 카메라별 RGB로 평가한 뒤 local affine을 적용하고 precomputed RGB를
splat한다. SplatAD 기본 경로는 [N,K,16] latent SH를 splat한 뒤 CNN decoder로
RGB를 만든다. latent feature에 Eq. (4)를 적용하면 논문과 다른 모델이 된다.

재현 기본 경로는 다음과 같이 고정한다.

1. ArmGS 실험에서는 latent CNN appearance branch를 끈다.
2. Gaussian color를 [N,K,3] RGB SH로 유지한다.
3. actor delta-SH를 적용한다.
4. 각 camera 방향에서 SH RGB를 외부 평가한다.
5. local affine 후 gsplat precomputed-color 경로로 전달한다.

기존 latent 경로를 유지하는 변형은 별도 ablation으로만 둔다.

### 2. actor/background ID를 명시적으로 변환한다

ArmGS group 규칙은 background=-1, actor>=0이다. SplatAD는 actor가
id < num_actors이고 static Gaussian이 id == num_actors이다. renderer 직전에
다음 변환을 수행한다.

    armgs_group_id = where(splatad_id < num_actors, splatad_id, -1)

ArmGS adapter의 기본 actor-alpha는 모든 actor를 하나의 indicator channel로
rasterize하므로 메모리가 actor 수에 따라 증가하지 않는다. actor별 채널은 작은
scene의 diagnostic mode로만 사용한다.

### 3. hash-grid는 visible Gaussian에만 적용한다

pure-PyTorch HashGridEncoder는 수식 검증용 reference이다. 현재 ArmGS pipeline은
conservative frustum pre-culling을 수행한 뒤 visible index에 대해서만 hash corner를
할당하고, 65,536 단위 chunking을 추가 적용한다.

RTX 4090 training forward/backward 측정에서 1M 전체 적용은 약 2.31 GiB,
1M scene 중 100k visible 적용은 약 261 MiB였다. 실제 논문 규모/FPS 재현에서는
tiny-cuda-nn 또는 fused gsplat-compatible encoder로 교체하되 reference 경로를
parity oracle로 유지한다.

## 인접 코드의 삽입 지점

기준으로 확인한 clean 인접 저장소는
/workspace/projects/camosplat/dohyun/CamoSplat_ECCV_2026 이다. 사용자 변경이 많은
CamoSplat_Pedestrian_deform worktree는 직접 수정하거나 통째로 복사하지 않는다.

- 설정/모듈 생성: neurad-studio/nerfstudio/models/splatad.py의
  SplatADModelConfig 및 populate_modules
- frame metadata: ad_dataparser.py에서 train/eval slicing 전에 global frame ID,
  sensor index, int64 timestamp 보존
- actor lifecycle: track timestamp 범위 밖 actor opacity를 0으로 두거나 scene에서 제외
- actor deformation: actor-local 좌표에서 pose 변환 전에 delta-mean/delta-SH 적용
- local refinement: world mean과 카메라별 SH RGB를 만든 뒤 rasterization 직전에 적용
- global refinement: RGB와 sky 합성 뒤 clamp 직전에 적용
- Eq. (9): splatad.py get_loss_dict에 depth/sky/foreground 항 연결
- optimizer: appearance, deformation, actor pose, sky parameter group 추가
- density control: 기존 strategy에 paper schedule(500–15000, interval 100) 설정

Nerfstudio camera forward는 -camera_to_world[...,2]를 사용하므로 CameraView에는
camera_convention=opengl을 전달한다. backend는 gsplat용 OpenCV view matrix로
명시적으로 변환한다. 새 view embedding은 동일 sensor index의 training frame 중
timestamp가 가장 가까운 row를 선택한다.

## 독립 패키지에서 이미 제공하는 계약

- single-camera [N,3] 및 batched per-camera [B,N,3] precomputed colors
- background/actor group mapping과 aggregate actor alpha
- cubemap sky lookup과 C_fg + (1-alpha) C_sky 합성
- OpenCV/OpenGL, wxyz quaternion, activated scale/opacity 계약
- float64 absolute timestamp 및 normalized actor conditioning
- track 범위 밖 actor 제외
- per-Gaussian velocity, camera linear/angular velocity, shutter time/direction forwarding
- Eq. (9) strict auxiliary availability 검증
- optimizer group, mean LR decay, model/optimizer/RNG checkpoint state

## 아직 필요한 SplatAD/data 연결

- 실제 SplatAD Gaussian ID를 ArmGS group ID로 변환하는 삽입 코드
- dataparser sky mask, projected LiDAR depth, actor bbox mask batch field
- actor/world velocity와 camera rolling-shutter metadata의 실제 값 생성
- LiDAR/COLMAP background 초기화 및 tracked-box actor canonical point 구축
- SplatAD density strategy와 optimizer-state migration
- dataset split, evaluator 및 training CLI

## 환경 주의사항

/venv/camosplat은 Torch 2.0.1+cu118, CUDA 11.8, gsplat, tinycudann,
Nerfstudio를 포함해 현재 adapter 테스트에 사용 가능하다. 다만 editable gsplat과
Nerfstudio가 사용자 변경이 있는 sibling worktree를 가리키므로, 전체 통합은 clean
baseline을 ArmGS 아래에 명시적으로 vendoring하거나 별도 clean worktree를 만든 뒤
진행해야 한다.
