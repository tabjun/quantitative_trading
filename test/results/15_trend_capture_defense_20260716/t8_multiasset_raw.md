# 15번 raw 결과 — t8_multiasset

- 실행 시각: 2026-07-16 18:58:10 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t8_multiasset
- 케이스 수: 4 (실패 0)

## 실행 인자

```json
{
  "suite": "t8_multiasset",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "ITransformerLike",
  "objectives": "huber,balanced_composite,directional_huber,variance_huber,correlation_huber,tail_huber,regime_huber,anti_collapse_v2",
  "normalizations": "window_standard,window_robust,asinh_revin",
  "preprocessings": "seasonal_diff16,winsor_025,none",
  "feature_sets": "coin_multitimeframe_structure",
  "seeds": "42",
  "gate_quantiles": "none,0.45,0.55,0.65",
  "objective": "huber",
  "normalization": "window_standard",
  "preprocessing": "seasonal_diff16",
  "horizon": 16,
  "risk_model": "PatchTSTLike",
  "risk_event_kind": "absolute_move",
  "event_quantile": 0.7,
  "seq_len": 64,
  "hidden": 96,
  "epochs": 10,
  "objective_warmup_epochs": 3,
  "patience": 4,
  "min_delta": 1e-05,
  "batch_size": 256,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "gradient_policy": "clip1",
  "lr": 0.001,
  "weight_decay": 0.0001,
  "max_rows": null,
  "max_windows": 24000,
  "stride": 2,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "cross_tickers": "",
  "ensemble_seeds": "42,7,123,2026",
  "tickers": "KRW-BTC,KRW-ETH,KRW-XRP,KRW-SOL",
  "ticker_tables": "KRW-BTC:btc_15m_advance,KRW-ETH:eth_15m_advance,KRW-XRP:xrp_15m_advance,KRW-SOL:sol_15m_advance",
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
  "memory_available_gb": 25.44489288330078,
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

## 이 suite가 보는 것 — 다자산 일반화

- **고정**: 우승 구성(모델·objective·전처리·h)을 고정한다.
- **변화**: 대상 종목(KRW-BTC/ETH/XRP/SOL)을 바꾼다.
- **질문**: 같은 구성의 신호가 BTC 전용인가, 다른 코인에도 일반화되는가?
- **읽는 법**: 여러 종목에서 trend_corr가 함께 양수면 신호가 구조적. 한 종목만이면 우연·과적합 의심.

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
        suite                   feature_set            model objective   preprocessing   normalization  horizon  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr        r2  direction_accuracy  large_move_da  momentum_da_baseline  momentum_large_da_baseline  variance_ratio  pred_return_std  actual_return_std  near_zero_share       mae_krw  persistence_mae_krw  copy_risk_krw  healthy_variance  naive_mae_zero  naive_mae_momentum  val_mase_momentum  val_trend_corr  val_large_move_da  val_variance_ratio  param_count  batch_size  epochs_run  best_val_loss  final_grad_norm  train_seconds  ticker   rows
t8_multiasset coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.005784        1.098776       0.713692    0.040986 -0.145253            0.514722       0.492222              0.455833                    0.474444        0.170782         0.003041           0.007359         0.212222 587888.914736        532931.666667       1.103123              True        0.005264            0.008105           0.706266        0.019385           0.515556            0.126039       155905         256          10       0.005761         0.456660            5.8 KRW-BTC 104750
t8_multiasset coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.007575        1.084161       0.691069    0.000624 -0.101424            0.496944       0.504444              0.454444                    0.428889        0.092840         0.003103           0.010183         0.269722  21082.055957         19365.000000       1.088668              True        0.006987            0.010961           0.689831       -0.002357           0.495556            0.072167       155905         256          10       0.007640         0.507475            5.9 KRW-ETH 104776
t8_multiasset coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.007635        1.086701       0.726448    0.004592 -0.125069            0.495278       0.486667              0.479722                    0.491111        0.107880         0.003234           0.009845         0.245556     13.917264            12.797500       1.087499              True        0.007026            0.010511           0.736562       -0.044063           0.485556            0.105391       155905         256          10       0.006901         0.472192            6.0 KRW-XRP 104767
t8_multiasset coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.009061        1.068573       0.699465   -0.024246 -0.108318            0.495000       0.467778              0.448611                    0.498889        0.093398         0.003587           0.011738         0.279167   1040.924771           971.944444       1.070971              True        0.008480            0.012954           0.724922        0.003988           0.498889            0.085987       155905         256          10       0.008301         0.417030            6.0 KRW-SOL 104802
```

## 종목별 일반화 요약

4종목 중 trend_corr>0 인 종목: 3. 과반이면 신호가 구조적, BTC만이면 종목 특정·과적합 의심.
