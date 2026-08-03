# ArmGS 논문 구현 분석

대상 논문: Guile Wu, Dongfeng Bai, Bingbing Liu, *ArmGS: Composite Gaussian
Appearance Refinement for Modeling Dynamic Urban Environments*, arXiv:2507.03886
(ICRA 2026).

## 1. 문제와 핵심 주장

기존 driving-scene 3DGS는 정적 배경과 object-centric 동적 actor를 합성해 빠르게
렌더링하지만, 프레임·카메라·시간에 따른 노출, 조명, 신호등/브레이크등, 국소
변형을 하나의 고정 appearance에 흡수한다. ArmGS는 이를 세 개의 서로 다른
granularity에서 보정한다.

1. **Local Gaussian level**: 각 Gaussian의 카메라별 RGB에 3채널 scale/bias를
   적용한다.
2. **Global image level**: rasterization 결과 전체에 3x3 color matrix와 3채널
   bias를 적용한다.
3. **Dynamic actor level**: 시간에 따라 actor Gaussian의 position과 SH 계수를
   변형한다.

세 보정은 기존 alpha compositing의 미분 가능성을 유지하며, 기하·appearance·actor
pose와 함께 end-to-end로 최적화된다.

## 2. 기준 3DGS와 composite scene

Gaussian 하나는 다음 파라미터를 가진다.

| 속성 | 형태 | 비고 |
|---|---:|---|
| mean `mu` | `[N, 3]` | world 또는 actor-local 좌표 |
| quaternion `r` | `[N, 4]` | covariance 회전 |
| scale `s` | `[N, 3]` | covariance 축 크기 |
| opacity `o` | `[N, 1]` | 보통 logit으로 최적화 |
| SH `h` | `[N, K, 3]` | `K=(degree+1)^2` |

픽셀은 깊이 순으로 정렬된 splat을

`C = sum_i c_i alpha_i product_{j<i}(1-alpha_j)`

로 합성한다. 장면은 세 부분으로 분해된다.

- **background**: world-coordinate Gaussians
- **dynamic actors**: actor별 object-coordinate Gaussians와 timestamp별 pose
- **sky**: view direction을 입력으로 하는 explicit cubemap Gaussian model

actor의 local mean/rotation은 학습 가능한 pose `(R_t, T_t)`를 통해
`mu_world = R_t mu_actor + T_t`, `r_world = R_t * r_actor`로 변환한다. 새 timestamp는
최적화된 keyframe pose 사이를 보간한다.

## 3. Local composite Gaussian refinement — Eq. (3), (4)

프레임 `q`마다 학습 가능한 저차원 embedding `epsilon_q`를 둔다. Gaussian 위치는
multi-resolution hash grid `H(mu)`로 부호화한다. 카메라 방향에 대해 SH를 평가한
현재 색 `c`와 연결하면

`f_l = concat(H(mu), epsilon_q, c)`

가 된다. 3-linear-layer ReLU MLP `D_l`은 Gaussian마다 `(alpha_l, beta_l)`, 각각
3채널을 출력하고

`c_local = alpha_l * c + beta_l`

를 rasterizer에 전달한다.

구현상 중요한 점:

- `c`는 raw SH가 아니라 해당 카메라 방향에서 평가된 RGB로 해석해야 Eq. (2)에
  바로 대입할 수 있다.
- background뿐 아니라 pose/deformation이 적용된 actor Gaussian도 composite set에
  합친 뒤 같은 local refiner를 통과시킨다.
- novel view에서는 동일 camera index의 training frame 중 timestamp가 가장 가까운
  embedding을 사용한다. 논문은 embedding 보간을 사용하지 않는다.
- MLP 마지막 층은 `alpha=1`, `beta=0`이 되도록 초기화해야 초기 3DGS rendering을
  훼손하지 않는다. 이는 논문에 명시되지 않은 안정화 선택이다.

## 4. Global image refinement — Eq. (5), (6)

카메라 위치와 view direction을 부호화한 viewpoint code `phi`를 frame embedding과
연결한다.

`f_g = concat(epsilon_q, phi)`

4-linear-layer ReLU MLP `D_g`가 image-wise `alpha_g [3,3]`, `beta_g [3]`를 예측하고,
모든 pixel에 동일한 affine transform을 적용한다.

`C_final = alpha_g @ C_rendered + beta_g`

코드에서는 channel-last row-vector convention을 사용하므로 수치 연산은
`C @ alpha_g.T + beta_g`가 된다. 마지막 층은 identity matrix와 zero bias로
초기화한다. 위치 정규화 범위, viewpoint encoding 종류/차원은 논문에 없으므로
scene AABB 정규화 + sinusoidal encoding을 설정 가능한 기본값으로 사용한다.

## 5. Dynamic actor refinement — Eq. (7), (8)

actor-local mean, timestamp, SH 전체를 연결한다.

`f_a = D_a(concat(F_a(mu_actor), F_f(t), h))`

- `F_a`: class-wise hash grid 또는 더 가벼운 sinusoidal position encoding
- `F_f`: timestamp sinusoidal encoding
- `D_a`: 2-linear-layer shared spatial-temporal encoder

multi-head 2-layer MLP `D_h`는 `delta_mu [N,3]`와 `delta_h [N,K,3]`를 따로 예측한다.

`mu_deformed = mu_actor + delta_mu`

`h_deformed = h + delta_h`

그 후 actor pose를 적용한다. 논문의 기본 실험은 가벼운 sinusoidal position
encoding을 쓴 것으로 해석한다. Implementation Details의 “F^f for dynamic actor
position encoding”은 Methodology에서 `F^f`를 time encoding으로 정의한 것과
충돌하므로 표기 오류 가능성이 높다. 첫 구현은 sinusoidal position/time encoding을
사용하고 class-wise hash/hypernetwork는 별도 옵션으로 남긴다.

## 6. 전체 forward 순서

1. 프레임/camera/timestamp를 조회하고 `epsilon`을 선택한다.
2. 각 actor의 local Gaussian에 time-conditioned `(delta_mu, delta_h)`를 적용한다.
3. timestamp pose를 보간해 actor-local Gaussian을 world 좌표로 옮긴다.
4. background와 actor의 SH를 각 Gaussian-to-camera 방향에서 RGB로 평가한다.
5. 모든 non-sky Gaussian에 local affine color refinement를 적용한다.
6. 하나의 깊이 정렬된 composite set으로 rasterize한다.
7. sky cubemap color를 residual transmittance와 합성한다.
8. 완성된 RGB에 global image affine을 적용한다.
9. RGB/DSSIM/depth/sky/foreground losses를 계산하고 모든 파라미터를 역전파한다.

actor와 background를 따로 렌더한 뒤 RGB를 합치면 서로의 occlusion ordering이
깨질 수 있으므로, 최종 splatting은 반드시 하나의 composite depth ordering을
사용해야 한다. actor alpha만 별도로 누적해 foreground loss용 auxiliary output을
얻는다.

## 7. Training objective — Eq. (9)

`L = (1-lambda1)L_rgb + lambda1 L_ssim + lambda2 L_depth + lambda3 L_sky + lambda4 L_fg`

- `L_rgb`: rendered/GT RGB L1
- `L_ssim`: `1 - SSIM` (3DGS 관례의 DSSIM)
- `L_depth`: valid projected LiDAR pixel에서 L1
- `L_sky`: rendered sky alpha와 사전 추출 sky mask의 BCE
- `L_fg`: actor accumulated alpha의 binary entropy

논문 기본값은 `lambda1=0.2`, `lambda2=0.01`, `lambda3=0.05`, `lambda4=0.1`이다.
관측치가 없는 auxiliary target은 그 항을 0으로 두되, 잘못된 silent omission을
막기 위해 trainer가 명시적으로 availability를 기록해야 한다.

## 8. 최적화와 density control

- Adam, 30,000 iterations
- mean LR: `1.6e-4 -> 1.6e-6`
- rotation LR: `1e-3`
- scale LR: `5e-3`
- opacity LR: `5e-2`
- SH LR: `2.5e-3`
- densification split/merge: iteration 500–15,000 사이 매 100 iteration
- local learner: 3 linear layers
- global learner: 4 linear layers
- class encoder: 2 linear layers
- actor encoder/head: 각각 2 linear layers

논문은 appearance MLP/embedding/hash-grid의 별도 LR을 주지 않는다. 따라서
parameter group을 분리하고 YAML에서 명시해야 한다.

## 9. 데이터 및 평가

| Dataset | protocol | 해상도 |
|---|---|---:|
| Waymo | 8 sequences, 매 4번째 frame test | 1066x1600 |
| KITTI | 3 sequences, 매 2번째 frame test | 375x1242 |
| NOTR | static-32 / dynamic-32 | 640x960 |
| VKITTI2 | 2 sequences, 50% dropout | 375x1242 |

초기 Gaussian은 LiDAR point cloud와 COLMAP SfM point를 사용한다. actor는 dataset의
tracked 3D boxes/tracklets로 초기화한다. 평가 지표는 PSNR, SSIM, LPIPS이고 actor
전용 평가는 projected 3D box mask 내부 PSNR을 쓴다.

## 10. 논문만으로 확정할 수 없는 항목

완전한 수치 재현 전에 다음 값은 저자 코드 또는 ablation이 필요하다.

- frame/class embedding dimension과 초기화
- local/global/actor MLP hidden width
- hash-grid level 수, feature 수, table size, resolution, AABB normalization
- camera viewpoint code의 정확한 positional encoding
- timestamp 정규화 범위
- global affine의 output constraint/clamping 및 RGB color space
- sky cubemap 구조와 해상도
- foreground entropy의 exact reduction/mask
- SSIM window/border 처리
- actor class-wise hypernetwork가 hash table weight를 생성하는 정확한 방식
- Gaussian split/merge threshold, opacity reset, pruning 규칙
- appearance/pose/sky optimizer LR과 scheduler
- 사용한 정확한 dataset sequence IDs와 preprocessing convention

따라서 첫 구현은 논문 수식과 tensor contract를 정확히 고정하고, 미공개 선택은
설정으로 격리하며, identity/zero initialization 테스트로 기존 3DGS baseline과의
호환성을 보장한다.

