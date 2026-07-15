# 15번 raw 결과 — t3_nonstationarity

- 실행 시각: 2026-07-16 02:31:57 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t3_nonstationarity
- 케이스 수: 18 (실패 0)

## 실행 인자

```json
{
  "suite": "t3_nonstationarity",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "ITransformerLike,PatchTSTLike",
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
  "max_windows": 16000,
  "stride": 4,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "cross_tickers": "",
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
  "memory_available_gb": 25.93130874633789,
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

## 지표 읽는 법 (요약)

- `mae_zero_ratio` < 1 이면 "0 예측(random-walk)" 기준선을 이긴 것.
- `mase_momentum` < 1 이면 "직전 h봉 추세 지속" 기준선을 이긴 것.
- `trend_corr`: 예측 vs 실제 h-step 수익률 Pearson 상관 (높을수록 추세 포착).
- `large_move_da`: |실제 수익률| 상위 25% 구간의 방향 정확도 (변동 큰 구간 포착력).
- `variance_ratio`: 예측분산/실제분산. ≪0.1 평탄화, ≫10 폭주, 0.05~20만 건강 판정.
- `copy_risk_krw`: KRW 스케일 MAE / persistence MAE. 1 미만이면 persistence보다 우수.

## 전체 leaderboard

```
             suite                   feature_set            model objective   preprocessing   normalization  horizon  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr        r2  direction_accuracy  large_move_da  momentum_da_baseline  momentum_large_da_baseline  variance_ratio  pred_return_std  actual_return_std  near_zero_share       mae_krw  persistence_mae_krw  copy_risk_krw  healthy_variance  naive_mae_zero  naive_mae_momentum  val_mase_momentum  val_trend_corr  val_large_move_da  val_variance_ratio  param_count  batch_size  epochs_run  best_val_loss  final_grad_norm  train_seconds  rank_score
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006213        1.204615       0.782249    0.071235 -0.291598            0.521250       0.523333                0.4625                       0.465        0.355765         0.004315           0.007234         0.126667 645956.893713            533893.75       1.209898              True        0.005158            0.007943           0.759075        0.031963           0.515000            0.174965       155905         256          10       0.006869         0.499846            4.6    0.616343
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16     asinh_revin       16    42    0.006206        1.203073       0.781247    0.041529 -0.289904            0.507083       0.513333                0.4625                       0.465        0.267795         0.003743           0.007234         0.128750 644402.639070            533893.75       1.206987              True        0.005158            0.007943           0.764156       -0.000864           0.495000            0.142116       155905         256          10       0.006907         0.549673            4.0    0.576737
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16   window_robust       16    42    0.006286        1.218630       0.791350    0.021898 -0.343100            0.507083       0.495000                0.4625                       0.465        0.359618         0.004338           0.007234         0.125000 652352.133042            533893.75       1.221876              True        0.005158            0.007943           0.774852        0.013544           0.490000            0.194745       155905         256          10       0.007009         0.457883            3.9    0.537764
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16 window_standard       16    42    0.005675        1.100214       0.714453   -0.010947 -0.147432            0.520417       0.501667                0.4625                       0.465        0.123523         0.002542           0.007234         0.209167 589331.601284            533893.75       1.103837              True        0.005158            0.007943           0.713968       -0.054372           0.460000            0.070439       162049         256          10       0.006397         0.209182            3.8    0.519274
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16     asinh_revin       16    42    0.005917        1.147181       0.744953   -0.014265 -0.215659            0.510000       0.491667                0.4625                       0.465        0.202040         0.003252           0.007234         0.182500 614104.165027            533893.75       1.150237              True        0.005158            0.007943           0.738313       -0.013099           0.486667            0.113834       162049         256          10       0.006637         0.429048            3.9    0.502907
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber            none     asinh_revin       16    42    0.006315        1.224222       0.794981   -0.012076 -0.345420            0.492083       0.488333                0.4625                       0.465        0.293771         0.003921           0.007234         0.131667 657643.970064            533893.75       1.231788              True        0.005158            0.007943           0.762040       -0.018335           0.485000            0.151834       155905         256          10       0.006897         0.505741            4.0    0.496759
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber      winsor_025   window_robust       16    42    0.006469        1.254220       0.814461   -0.012348 -0.388459            0.490000       0.490000                0.4625                       0.465        0.336214         0.004194           0.007234         0.134583 674502.843811            533893.75       1.263365              True        0.005158            0.007943           0.772357       -0.013388           0.493333            0.175327       155905         256          10       0.006992         0.502029            3.7    0.496206
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber      winsor_025     asinh_revin       16    42    0.005823        1.128867       0.733060   -0.013877 -0.182387            0.501667       0.483333                0.4625                       0.465        0.148355         0.002786           0.007234         0.208750 606201.972191            533893.75       1.135436              True        0.005158            0.007943           0.727877       -0.033416           0.456667            0.081896       162049         256          10       0.006528         0.234215            4.0    0.496150
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber      winsor_025     asinh_revin       16    42    0.006392        1.239273       0.804755   -0.006934 -0.366932            0.482500       0.476667                0.4625                       0.465        0.301519         0.003972           0.007234         0.131250 665964.259642            533893.75       1.247372              True        0.005158            0.007943           0.770190       -0.034678           0.500000            0.159677       155905         256          10       0.006988         0.446164            4.0    0.489257
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber            none   window_robust       16    42    0.006396        1.239919       0.805175   -0.016873 -0.378412            0.491250       0.481667                0.4625                       0.465        0.309944         0.004027           0.007234         0.141667 666569.509210            533893.75       1.248506              True        0.005158            0.007943           0.769259       -0.023490           0.495000            0.163114       155905         256          10       0.006942         0.492589            3.9    0.484276
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber            none     asinh_revin       16    42    0.005834        1.131023       0.734460   -0.016687 -0.176089            0.488750       0.473333                0.4625                       0.465        0.161430         0.002906           0.007234         0.211667 606656.766748            533893.75       1.136287              True        0.005158            0.007943           0.730730       -0.033126           0.471667            0.094858       162049         256          10       0.006547         0.218184            4.0    0.483200
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber            none window_standard       16    42    0.006448        1.250032       0.811741   -0.017735 -0.404957            0.499167       0.476667                0.4625                       0.465        0.293728         0.003920           0.007234         0.142083 671517.901726            533893.75       1.257774              True        0.005158            0.007943           0.770473       -0.022449           0.473333            0.150493       155905         256          10       0.006983         0.496847            3.9    0.477757
t3_nonstationarity coin_multitimeframe_structure ITransformerLike     huber      winsor_025 window_standard       16    42    0.006264        1.214401       0.788604   -0.018740 -0.336816            0.489167       0.470000                0.4625                       0.465        0.276089         0.003801           0.007234         0.152917 652307.875135            533893.75       1.221793              True        0.005158            0.007943           0.760679       -0.025262           0.480000            0.148169       155905         256          10       0.006888         0.439799            4.0    0.472399
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber            none   window_robust       16    42    0.005967        1.156837       0.751223   -0.021472 -0.231740            0.493750       0.461667                0.4625                       0.465        0.169025         0.002974           0.007234         0.185000 621764.370116            533893.75       1.164584              True        0.005158            0.007943           0.734827       -0.014245           0.466667            0.090487       162049         256          10       0.006607         0.323536            3.9    0.465072
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber      winsor_025 window_standard       16    42    0.005710        1.106932       0.718816   -0.031319 -0.156161            0.472083       0.450000                0.4625                       0.465        0.120318         0.002509           0.007234         0.220417 593492.565838            533893.75       1.111630              True        0.005158            0.007943           0.711859        0.003236           0.481667            0.065994       162049         256          10       0.006367         0.293010            4.0    0.446799
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16   window_robust       16    42    0.006317        1.224747       0.795322   -0.027945 -0.360441            0.496250       0.453333                0.4625                       0.465        0.313626         0.004051           0.007234         0.138333 658769.965792            533893.75       1.233897              True        0.005158            0.007943           0.768255        0.022562           0.473333            0.172370       162049         256          10       0.006927         0.403032            3.9    0.445856
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber            none window_standard       16    42    0.005748        1.114301       0.723601   -0.042763 -0.172461            0.487083       0.453333                0.4625                       0.465        0.114210         0.002445           0.007234         0.215417 597703.723592            533893.75       1.119518              True        0.005158            0.007943           0.715154        0.008059           0.495000            0.064192       162049         256          10       0.006410         0.238842            3.9    0.438210
t3_nonstationarity coin_multitimeframe_structure     PatchTSTLike     huber      winsor_025   window_robust       16    42    0.005977        1.158823       0.752513   -0.031628 -0.221910            0.476667       0.441667                0.4625                       0.465        0.177525         0.003048           0.007234         0.195417 622150.797215            533893.75       1.165308              True        0.005158            0.007943           0.738220       -0.026794           0.460000            0.100907       162049         256          10       0.006631         0.251139            3.7    0.434787
```

## normalization별 평균

```
  normalization  trend_corr  large_move_da  mase_momentum  mae_zero_ratio  variance_ratio  direction_accuracy
    asinh_revin     -0.0037         0.4878         0.7656          1.1789          0.2292              0.4970
  window_robust     -0.0147         0.4706         0.7850          1.2089          0.2777              0.4925
window_standard     -0.0084         0.4792         0.7566          1.1651          0.2139              0.4982
```
