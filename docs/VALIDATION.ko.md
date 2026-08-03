# ArmGS 구현 검증 현황

검증일: 2026-08-03

## 사용자 지적 사항 처리 결과

| 항목 | 상태 | 구현/검증 |
|---|---|---|
| background/actor/sky/trainer 부재 | 해결 | learnable composite scene, cubemap sky, paper-ordered renderer, one-view trainer |
| 통합 순서와 gradient 미검증 | 해결 | exact CPU ordering test와 실제 gsplat CUDA Eq. (9) backward |
| 전체 Gaussian hash-grid 메모리 | 해결 경로 구현 | conservative frustum selection + visible/chunked hash API |
| frustum/backend visibility | 해결 | near/far/eps2d 동기화와 CUDA edge parity; nonzero rolling shutter는 culling 해제 |
| SH backend parity | 해결 | 공식 3DGS CPU와 gsplat CPU/CUDA degree 0–3 비교 |
| 좌표/파라미터 convention | 해결 | OpenCV/OpenGL 변환, wxyz quaternion, activated renderer contract, raw learnable wrapper |
| near-zero quaternion | 해결 | 초기값과 매 forward에서 거부해 invisible splat/NaN gradient 방지 |
| actor별 H×W×A alpha | 해결 | 기본 aggregate O(HW), per-group은 명시적 진단 옵션 |
| actor alpha 지원 부재 | 해결 | active actor의 alpha=None을 숨기지 않고 strict foreground loss에서 실패 |
| actor lifecycle | 해결 | track 범위 밖 actor는 composite에서 제외하고 alpha zero map 제공 |
| timestamp aliasing/정밀도 | 해결 | module dtype cast 후에도 absolute time float64 보존, conditioning만 정규화 |
| affine 폭주 | 완화 | YAML에서 켜는 identity-centered tanh bounds |
| checkpoint/LR 안전성 | 해결 | CUDA map-location/GPU 수 변화 RNG 복원과 finite-positive LR 강제 |
| SSIM/depth 정의 불명확 | 가정 격리 | SSIM window/sigma/range와 gsplat D/ED 선택을 설정으로 노출 |
| class-wise actor hash | 미구현 | 기본 sinusoidal 경로에는 불필요하며 관련 ablation만 남음 |

## 수치 검증

SH 4,096개 무작위 샘플의 최대 절대 오차:

| backend | degree 0–3 최대 오차 범위 |
|---|---:|
| 공식 CPU reference | 2.22e-16 – 1.78e-15 |
| gsplat CUDA | 0 – 1.43e-6 |

Hash-grid training forward/backward benchmark, RTX 4090, 8 levels × 2 features,
chunk size 65,536:

| 전체 Gaussian | visible/hash 적용 | peak allocated | elapsed |
|---:|---:|---:|---:|
| 100,000 | 100,000 | 241.6 MiB | 207.9 ms |
| 1,000,000 | 1,000,000 | 2,313.1 MiB | 142.9 ms |
| 1,000,000 | 100,000 | 260.7 MiB | 229.9 ms |

시간은 CUDA warm-up 순서에 민감하므로 메모리 비교가 주 목적이다.

전체 테스트:

~~~text
74 passed, 4 warnings
~~~

경고 4개는 설치된 gsplat 환경의 pkg_resources deprecation이며 ArmGS 실패가
아니다.

## 아직 남은 blocker

현재 코드는 독립 synthetic scene를 실제 gsplat로 학습할 수 있지만, 다음 항목이
없어 논문 데이터셋의 최종 수치를 바로 재현할 수는 없다.

1. KITTI/Waymo camera, LiDAR, tracklet/box converter
2. LiDAR/COLMAP 기반 Gaussian 초기화
3. 논문에 공개되지 않은 densification/pruning/opacity reset 세부 규칙
4. SplatAD rolling-shutter velocity/shutter metadata 연결
5. class-wise actor hash-grid/hypernetwork ablation
6. dataset evaluator와 논문 표 자동 비교 CLI

특히 density-control threshold는 논문만으로 확정할 수 없으므로 임의의 값을
“논문 설정”으로 표시하지 않고, clean baseline 전략을 선택한 구현 가정으로
기록해야 한다.
