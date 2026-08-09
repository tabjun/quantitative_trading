# 15번 raw 결과 — t9_bridge

- 실행 시각: 2026-07-16 18:33:07 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t9_bridge
- 케이스 수: 8 (실패 0)

## 실행 인자

```json
{
  "suite": "t9_bridge",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "Linear,PatchTSTLike",
  "objectives": "huber,balanced_composite,directional_huber,variance_huber,correlation_huber,tail_huber,regime_huber,anti_collapse_v2",
  "normalizations": "window_standard,window_robust,asinh_revin",
  "preprocessings": "seasonal_diff16,winsor_025,none",
  "feature_sets": "coin_multitimeframe_structure",
  "seeds": "42,2026",
  "gate_quantiles": "none,0.45,0.55,0.65",
  "objective": "balanced_composite",
  "normalization": "window_standard",
  "preprocessing": "seasonal_diff16",
  "horizon": 16,
  "risk_model": "PatchTSTLike",
  "risk_event_kind": "absolute_move",
  "event_quantile": 0.7,
  "seq_len": 64,
  "hidden": 96,
  "epochs": 12,
  "objective_warmup_epochs": 3,
  "patience": 5,
  "min_delta": 1e-05,
  "batch_size": 48,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "gradient_policy": "clip1",
  "lr": 0.001,
  "weight_decay": 0.0001,
  "max_rows": null,
  "max_windows": 4096,
  "stride": 1,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "cross_tickers": "",
  "ensemble_seeds": "42,7,123,2026",
  "tickers": "KRW-BTC,KRW-ETH,KRW-XRP,KRW-SOL",
  "ticker_tables": "",
  "bridge_rows": 40000,
  "bridge_periods": "early,recent",
  "cost_bps": 14.0,
  "policy_entry_bps": 20.0,
  "min_active_share": 0.05,
  "min_trade_count": 5,
  "dry_run": false,
  "continue_on_failure": true
}
```

## 환경

```json
{
  "python_executable": "/home/std_jun99120/personal_ai_project/quantitative_trading/.venvs/quant_uv_py312_20260714_005036/bin/python",
  "python_version": "3.12.13 (main, Jun 23 2026, 15:18:55) [Clang 22.1.3 ]",
  "platform": "Linux-7.0.0-27-generic-x86_64-with-glibc2.43",
  "cpu_logical": 32,
  "cpu_physical": 16,
  "memory_total_gb": 30.414440155029297,
  "memory_available_gb": 25.393325805664062,
  "torch_version": "2.10.0+cu126",
  "torch_cuda_version": "12.6",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_gb": 23.5096435546875,
  "selected_resource_profile": "school_4090_15gb",
  "applied_n_jobs": 16,
  "applied_max_workers": 16,
  "applied_num_workers": 0,
  "applied_torch_threads": 16,
  "applied_torch_interop_threads": 4,
  "applied_optuna_n_jobs": 1,
  "applied_batch_size": 48
}
```

## 데이터 기초 통계

```json
{
  "rows": 104750,
  "start": "2023-07-18 04:45:00",
  "end": "2026-07-16 01:45:00",
  "close_mean": 105976068.07637231,
  "close_std": 38525514.776594825,
  "close_min": 34171000.0,
  "close_max": 179810000.0,
  "return_mean": 8.686274754900732e-06,
  "return_std": 0.0023861262170528346,
  "return_skew": 0.0037859150748292777,
  "return_kurtosis": 107.21892931168455,
  "missing_cells": 0
}
```

## 이 suite가 보는 것 — 12번 챔피언 브리지 재현 — 진동 폭 축소 원인 분리

- **고정**: 12번 case 41과 완전히 동일한 설정: 1-step(다음 15분) target, Linear(+비교군), balanced_composite, seasonal_diff16, window_standard, seq_len 64, max_windows 4096, stride 1, epochs 12, patience 5, batch 48(=school_4090_15gb profile).
- **변화**: 데이터 구간만 바꾼다 — early(DB 앞 40k행 ≈ 12번과 같은 2023~2024 구간) vs recent(DB 뒤 40k행 ≈ 2025~2026 최근 구간).
- **질문**: 예전(12번)에 보이던 '변동을 유지하며 추세를 따라가는 예측'이 같은 설정으로 재현되는가? 지금의 낮은 진동 폭이 설정 탓인가, 시장 구간 탓인가, h-step 전환 탓인가?
- **읽는 법**: early에서 variance_ratio·Pearson이 12번 기록(var≈0.1~0.15, Pearson≈0.13)에 근접하면 설정 재현 성공. recent에서만 죽으면 구간(레짐) 문제. 둘 다 죽으면 데이터/엔진 차이를 더 파야 한다. 그림은 12번 case 41 진단(return prediction/next-candle/scatter)과 같은 구성으로 저장하니 직접 나란히 비교한다.

## 지표 사전 (각 컬럼이 무엇인가)

- `mae_return`: 예측 h-step 누적수익률의 평균절대오차. 낮을수록 좋지만 절대값만으로는 판단 불가(아래 비율로 봄).
- `mae_zero_ratio`: 예측 오차 ÷ '항상 0(random-walk)으로 예측' 오차. <1이면 랜덤워크 기준선을 이긴 것, ≥1이면 못 이긴 것.
- `mase_momentum`: 예측 오차 ÷ '직전 h봉 추세가 지속된다고 가정' 오차. <1이면 단순 추세지속 기준선을 이긴 것.
- `trend_corr`: 예측 vs 실제 h-step 수익률의 Pearson 상관. 0=무상관, 높을수록 추세 방향·크기 동조. 금융 시계열에선 0.1도 유의미.
- `r2`: 결정계수(설명력). 0 이상이어야 평균 예측보다 나음. 음수면 '그냥 평균/0을 찍는 것보다 못하다'는 뜻.
- `direction_accuracy`: 전체 구간 방향(부호) 정확도. 0.5=동전던지기.
- `large_move_da`: |실제 수익률| 상위 25%(큰 변동) 구간의 방향 정확도. 사용자가 중시하는 '큰 변동 포착력'. 0.5 초과면 우위.
- `variance_ratio`: 예측분산 ÷ 실제분산. ≪0.1=평탄화(변동 죽음), ≫20=폭주, 0.05~20을 healthy로 판정. 1 근처가 이상적.
- `copy_risk_krw`: KRW 스케일 MAE ÷ persistence(직전값) MAE. <1이면 직전값 복사보다 우수.
- `healthy_variance`: variance_ratio가 0.05~20 안에 있으면 True. 평탄화/폭주 둘 다 아님.
- `qualified`: (T5) active_share≥하한 && trade_count≥하한. '거의 거래 안 해서 MDD가 좋아 보이는' 케이스를 우승에서 제외.
- `mdd`: (T5) 최대낙폭. 자산곡선이 고점 대비 가장 많이 빠진 비율. 0에 가까울수록 방어 우수.

## 전체 leaderboard

```
    suite period                               period_span        model          objective   preprocessing   normalization  seed         target       mae_krw  persistence_mae_krw  copy_risk_krw  direction_accuracy  variance_ratio  pred_return_std  actual_return_std   pearson  near_zero_share  epochs_run
t9_bridge  early 2023-07-18 04:45:00 ~ 2024-09-06 14:45:00       Linear balanced_composite seasonal_diff16 window_standard    42 1-step(다음 15분) 166332.589445        117482.926829       1.415802            0.544715        0.838620         0.001919           0.002095  0.026467         0.099187          12
t9_bridge  early 2023-07-18 04:45:00 ~ 2024-09-06 14:45:00       Linear balanced_composite seasonal_diff16 window_standard  2026 1-step(다음 15분) 164694.129434        117482.926829       1.401856            0.513821        0.747262         0.001811           0.002095  0.001271         0.112195          12
t9_bridge  early 2023-07-18 04:45:00 ~ 2024-09-06 14:45:00 PatchTSTLike balanced_composite seasonal_diff16 window_standard    42 1-step(다음 15분) 209670.135551        117482.926829       1.784686            0.560976        1.726331         0.002753           0.002095  0.028699         0.069919          12
t9_bridge  early 2023-07-18 04:45:00 ~ 2024-09-06 14:45:00 PatchTSTLike balanced_composite seasonal_diff16 window_standard  2026 1-step(다음 15분) 201323.542606        117482.926829       1.713641            0.512195        0.868541         0.001953           0.002095 -0.041982         0.065041          12
t9_bridge recent 2025-05-24 07:30:00 ~ 2026-07-16 01:45:00       Linear balanced_composite seasonal_diff16 window_standard    42 1-step(다음 15분) 157954.009720        102959.349593       1.534140            0.502439        0.946487         0.001492           0.001533  0.008778         0.078049          12
t9_bridge recent 2025-05-24 07:30:00 ~ 2026-07-16 01:45:00       Linear balanced_composite seasonal_diff16 window_standard  2026 1-step(다음 15분) 182904.800745        102959.349593       1.776476            0.510569        1.650972         0.001970           0.001533  0.001735         0.074797          12
t9_bridge recent 2025-05-24 07:30:00 ~ 2026-07-16 01:45:00 PatchTSTLike balanced_composite seasonal_diff16 window_standard    42 1-step(다음 15분) 211157.803045        102959.349593       2.050885            0.487805        2.131076         0.002238           0.001533 -0.016292         0.037398          12
t9_bridge recent 2025-05-24 07:30:00 ~ 2026-07-16 01:45:00 PatchTSTLike balanced_composite seasonal_diff16 window_standard  2026 1-step(다음 15분) 176790.926931        102959.349593       1.717094            0.533333        1.545494         0.001906           0.001533  0.056722         0.078049          12
```

## 비교 기준 (과거 기록)

12번 case 41 기록(2026-06-24, 40k행/4096윈도우/epochs12): Pearson 0.129, fusion return +1.79% (96케이스 중 유일 양수). 10번 동계열(Linear+balanced_composite): copy_risk 1.06~1.13, variance_ratio 0.10~0.17, DA 51~53%. 14번(12k행/2048윈도우): 같은 구성이 copy_risk 67.6, variance_ratio 3679로 폭주 — 규모 아티팩트 의혹. 이번 T9가 그 세 기록 사이 어디에 앉는지가 판정 기준.
