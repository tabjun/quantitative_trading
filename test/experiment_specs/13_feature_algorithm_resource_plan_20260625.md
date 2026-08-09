# 13번 feature·algorithm·resource 공동 최적화 계획

## 목적

12번에서 가장 유망했던 multi-timeframe 구조가 실제로 좋은 독립변수군인지 확인하기 위해, 같은 변수셋을 내부 분해하고 알고리즘·전처리·risk gate·seed 안정성을 함께 넓게 본다.

이 실험은 단순 feature group ablation이 아니다. 변수 후보군이 특정 seed나 특정 모델에서만 운 좋게 좋아지는지, 아니면 여러 모델·전처리·위험 게이트 조건에서도 버티는지를 확인하는 대규모 본실험이다.

## 보호 규칙

- 완료된 12번 노트북은 이번 작업에서 읽기 전용 source of truth로만 사용한다.
- 13번 결과가 필요하면 새 번호인 `13_feature_algorithm_resource_test`에 누적한다.
- 보고서 작성 시 완료된 13번 노트북도 strip, clear output, re-save, 코드 셀 재작성 대상으로 삼지 않는다.

## 실험 축

| 축 | 값 |
|---|---|
| multi-timeframe 분해 | returns only, volatility only, trend/position only, volume/range only, returns+volatility, returns+trend, volatility+trend, full |
| 변수 확장 | momentum/reversal, calendar cycle, shock event, attention proxy, liquidity micro, order-flow proxy, full available, optional text/cross-market/macro/on-chain/derivatives |
| 점예측 모델 | Linear, PatchTSTLike, TCNLike, ModernTCNLike, DLinearLike, NLinearLike, TransformerLike, ITransformerLike, AutoformerLike, MambaLike |
| risk 모델 | 기본 PatchTSTLike, 확장 Linear/TCNLike/PatchTSTLike |
| 전처리 | seasonal_diff16, winsor_025, frequency_bandpass, median_residual_5, linear_detrend+asinh_robust, volatility_scale+asinh_robust |
| risk gate | horizon 8/16/32, absolute_move/downside, allow quantile 0.45/0.55/0.65 |
| seed | 42, 137, 2026, 777, 4096 |

## suite

| suite | 역할 |
|---|---|
| `dry_plan` | 실행 가능한 feature/model/preprocessing/risk/seed case 수와 optional group 제외 사유만 출력 |
| `mtf_decomposition` | 12번 1순위 multi-timeframe 변수셋의 내부 구성요소 비교 |
| `algorithm_screen` | 좁힌 변수셋에서 알고리즘을 넓게 비교 |
| `feature_algorithm_matrix` | 변수 후보군과 알고리즘을 함께 비교 |
| `risk_gate_sensitivity` | risk horizon, event kind, cutoff 민감도 비교 |
| `full_resource` | 서버 독점 자원 기준 전체 대형 matrix |

## 자원 기본값

- `profile=school_4090_15gb`
- `parallel_slot=exclusive`
- `device=cuda`
- `epochs=24`
- `patience=7`
- `seq_len=64`, `full_resource`에서는 `128`도 추가 확인
- `hidden=128`, OOM 시 case 재시도에서 `96` fallback
- `batch_size=1024`
- `--large-batch 2048`은 `full_resource`에서 선택
- OOM batch downshift는 backend에서 batch size를 절반으로 줄이며 실제 사용 batch size를 결과표에 남긴다.

## 실행 예시

```python
%run test/models/13_feature_algorithm_resource_test.py --suite dry_plan --max-rows 2000
%run test/models/13_feature_algorithm_resource_test.py --suite mtf_decomposition --batch-size 1024
%run test/models/13_feature_algorithm_resource_test.py --suite algorithm_screen --batch-size 1024 --continue-on-failure
%run test/models/13_feature_algorithm_resource_test.py --suite feature_algorithm_matrix --batch-size 1024 --continue-on-failure
%run test/models/13_feature_algorithm_resource_test.py --suite risk_gate_sensitivity --batch-size 1024 --continue-on-failure
%run test/models/13_feature_algorithm_resource_test.py --suite full_resource --large-batch 2048 --continue-on-failure
```

## 판정 기준

1. `fusion MDD`가 덜 음수이면서 거래가 너무 적지 않아야 한다.
2. `fusion return after cost`가 양수 또는 손실 축소 방향이어야 한다.
3. `copy-risk ratio`가 낮아져야 하며, 최소한 12번 후보보다 악화되지 않아야 한다.
4. risk branch는 AP lift, Brier skill, ECE가 함께 무너지지 않아야 한다.
5. 같은 feature group이 seed 하나에서만 좋으면 최종 후보로 승격하지 않는다.
