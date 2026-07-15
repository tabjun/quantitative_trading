# 15번 raw 결과 — t4_feature

- 실행 시각: 2026-07-16 02:33:13 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t4_feature
- 케이스 수: 16 (실패 0)

## 실행 인자

```json
{
  "suite": "t4_feature",
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
  "feature_sets": "coin_multitimeframe_structure,mtf_returns,mtf_volatility,mtf_trend,mtf_volume_range,mtf_plus_momentum,mtf_plus_shock,mtf_plus_volregime",
  "seeds": "42,7",
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
  "memory_available_gb": 25.871410369873047,
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
t4_feature coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006213        1.204615       0.782249    0.071235 -0.291598            0.521250       0.523333                0.4625                       0.465        0.355765         0.004315           0.007234         0.126667 645956.893713            533893.75       1.209898              True        0.005158            0.007943           0.759075        0.031963           0.515000            0.174965       155905         256          10       0.006869         0.499846            4.6    0.616343
t4_feature                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006033        1.169660       0.759550    0.063103 -0.229844            0.497083       0.495000                0.4625                       0.465        0.195535         0.003199           0.007234         0.092083 627543.438349            533893.75       1.175409              True        0.005158            0.007943           0.759112       -0.021059           0.476667            0.119139       155905         256          10       0.006849         0.636269            2.2    0.582148
t4_feature             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.006266        1.214782       0.788851    0.025647 -0.324558            0.524167       0.533333                0.4625                       0.465        0.339837         0.004217           0.007234         0.132083 651709.361879            533893.75       1.220672              True        0.005158            0.007943           0.763646       -0.011158           0.503333            0.182842       155905         256          10       0.006929         0.492052            1.6    0.580096
t4_feature            mtf_plus_volregime ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.006125        1.187462       0.771110    0.050749 -0.264000            0.510417       0.501667                0.4625                       0.465        0.320629         0.004096           0.007234         0.147917 637164.874482            533893.75       1.193430              True        0.005158            0.007943           0.768009       -0.001251           0.521667            0.179674       155905         256          10       0.006952         0.485131            1.6    0.575305
t4_feature                mtf_plus_shock ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.006261        1.213821       0.788227    0.029740 -0.334903            0.514167       0.523333                0.4625                       0.465        0.361052         0.004347           0.007234         0.136667 651198.444608            533893.75       1.219715              True        0.005158            0.007943           0.776300       -0.027923           0.463333            0.181205       155905         256          10       0.006997         0.433777            1.6    0.574250
t4_feature                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.005655        1.096309       0.711917    0.033529 -0.125733            0.497083       0.505000                0.4625                       0.465        0.120793         0.002514           0.007234         0.233750 587206.559421            533893.75       1.099857              True        0.005158            0.007943           0.711596       -0.045856           0.466667            0.060621       155905         256          10       0.006358         0.826021            1.8    0.567337
t4_feature             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006421        1.244833       0.808366    0.038842 -0.363794            0.497083       0.501667                0.4625                       0.465        0.279519         0.003824           0.007234         0.122500 668571.335962            533893.75       1.252255              True        0.005158            0.007943           0.771775        0.017301           0.501667            0.149906       155905         256          10       0.006992         0.712835            1.6    0.559672
t4_feature            mtf_plus_volregime ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006125        1.187366       0.771047    0.034378 -0.272731            0.501250       0.500000                0.4625                       0.465        0.296999         0.003942           0.007234         0.146250 637228.855019            533893.75       1.193550              True        0.005158            0.007943           0.743411        0.029440           0.500000            0.151045       155905         256          10       0.006701         0.419238            1.8    0.557274
t4_feature                mtf_volatility ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006264        1.214479       0.788654    0.005647 -0.340873            0.514583       0.520000                0.4625                       0.465        0.297531         0.003946           0.007234         0.158750 652049.228061            533893.75       1.221309              True        0.005158            0.007943           0.757789        0.038320           0.505000            0.157624       155905         256          10       0.006839         0.677811            1.9    0.546781
t4_feature                mtf_plus_shock ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006414        1.243443       0.807463    0.024514 -0.366974            0.491250       0.501667                0.4625                       0.465        0.339394         0.004214           0.007234         0.132917 667288.449210            533893.75       1.249853              True        0.005158            0.007943           0.775248        0.018897           0.501667            0.177731       155905         256          10       0.007014         0.423362            1.6    0.545435
t4_feature coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.006218        1.205519       0.782835    0.014940 -0.310077            0.514583       0.503333                0.4625                       0.465        0.319569         0.004089           0.007234         0.147917 646500.226224            533893.75       1.210916              True        0.005158            0.007943           0.762462       -0.021164           0.518333            0.167836       155905         256          10       0.006895         0.591047            4.1    0.539990
t4_feature                   mtf_returns ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.005850        1.134228       0.736541    0.000172 -0.185041            0.497083       0.476667                0.4625                       0.465        0.063488         0.001823           0.007234         0.121667 608526.764716            533893.75       1.139790              True        0.005158            0.007943           0.721924       -0.043812           0.473333            0.034694       155905         256          10       0.006458         0.809327            3.8    0.503185
t4_feature                mtf_volatility ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.006170        1.196275       0.776833    0.007715 -0.280446            0.489583       0.471667                0.4625                       0.465        0.224837         0.003430           0.007234         0.146667 643026.087792            533893.75       1.204408              True        0.005158            0.007943           0.743628       -0.005216           0.463333            0.109290       155905         256          10       0.006684         0.606066            1.6    0.501698
t4_feature                   mtf_returns ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006225        1.206894       0.783729   -0.003646 -0.298856            0.489167       0.480000                0.4625                       0.465        0.231423         0.003480           0.007234         0.105833 647661.001540            533893.75       1.213090              True        0.005158            0.007943           0.739471        0.001107           0.480000            0.123947       155905         256          10       0.006652         0.594162            3.9    0.497982
t4_feature              mtf_volume_range ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.006387        1.238330       0.804143   -0.005922 -0.369754            0.493333       0.483333                0.4625                       0.465        0.232418         0.003487           0.007234         0.130833 664811.292448            533893.75       1.245213              True        0.005158            0.007943           0.769694        0.019473           0.510000            0.141175       155905         256          10       0.006954         0.971713            1.5    0.496997
t4_feature              mtf_volume_range ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.005925        1.148600       0.745874   -0.021098 -0.213090            0.487500       0.470000                0.4625                       0.465        0.119344         0.002499           0.007234         0.174167 616269.124032            533893.75       1.154292              True        0.005158            0.007943           0.727482       -0.005861           0.488333            0.068530       155905         256          10       0.006519         0.678998            1.5    0.474314
```

## feature_set별 평균

```
                  feature_set  trend_corr  large_move_da  mase_momentum  mae_zero_ratio  variance_ratio  direction_accuracy
coin_multitimeframe_structure      0.0431         0.5133         0.7825          1.2051          0.3377              0.5179
            mtf_plus_momentum      0.0322         0.5175         0.7986          1.2298          0.3097              0.5106
               mtf_plus_shock      0.0271         0.5125         0.7978          1.2286          0.3502              0.5027
           mtf_plus_volregime      0.0426         0.5008         0.7711          1.1874          0.3088              0.5058
                  mtf_returns     -0.0017         0.4783         0.7601          1.1706          0.1475              0.4931
                    mtf_trend      0.0483         0.5000         0.7357          1.1330          0.1582              0.4971
               mtf_volatility      0.0067         0.4958         0.7827          1.2054          0.2612              0.5021
             mtf_volume_range     -0.0135         0.4767         0.7750          1.1935          0.1759              0.4904
```
