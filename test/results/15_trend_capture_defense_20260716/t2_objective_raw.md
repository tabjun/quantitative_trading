# 15번 raw 결과 — t2_objective

- 실행 시각: 2026-07-16 02:28:31 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t2_objective
- 케이스 수: 24 (실패 0)

## 실행 인자

```json
{
  "suite": "t2_objective",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "ITransformerLike,PatchTSTLike,ModernTCNLike",
  "objectives": "huber,balanced_composite,directional_huber,variance_huber,correlation_huber,tail_huber,regime_huber,anti_collapse_v2",
  "normalizations": "window_standard,window_robust,asinh_revin",
  "preprocessings": "seasonal_diff16,winsor_025,none",
  "feature_sets": "coin_multitimeframe_structure",
  "seeds": "42",
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
  "memory_available_gb": 25.94152069091797,
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
       suite                   feature_set            model          objective   preprocessing   normalization  horizon  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr         r2  direction_accuracy  large_move_da  momentum_da_baseline  momentum_large_da_baseline  variance_ratio  pred_return_std  actual_return_std  near_zero_share      mae_krw  persistence_mae_krw  copy_risk_krw  healthy_variance  naive_mae_zero  naive_mae_momentum  val_mase_momentum  val_trend_corr  val_large_move_da  val_variance_ratio  param_count  batch_size  epochs_run  best_val_loss  final_grad_norm  train_seconds  rank_score
t2_objective coin_multitimeframe_structure ITransformerLike              huber seasonal_diff16 window_standard       16    42    0.006213        1.204615       0.782249    0.071235  -0.291598            0.521250       0.523333                0.4625                       0.465        0.355765         0.004315           0.007234         0.126667 6.459569e+05            533893.75       1.209898              True        0.005158            0.007943           0.759075        0.031963           0.515000            0.174965       155905         256          10       0.006869         0.499846            4.5    0.616343
t2_objective coin_multitimeframe_structure ITransformerLike         tail_huber seasonal_diff16 window_standard       16    42    0.006227        1.207252       0.783961    0.068515  -0.303889            0.516667       0.521667                0.4625                       0.465        0.314099         0.004054           0.007234         0.123333 6.475280e+05            533893.75       1.212841              True        0.005158            0.007943           0.767926        0.028786           0.500000            0.160632       155905         256          10       0.009016         0.575783            4.1    0.611786
t2_objective coin_multitimeframe_structure ITransformerLike  directional_huber seasonal_diff16 window_standard       16    42    0.006382        1.237244       0.803437    0.055875  -0.359115            0.508750       0.516667                0.4625                       0.465        0.333155         0.004175           0.007234         0.117500 6.644346e+05            533893.75       1.244507              True        0.005158            0.007943           0.774524        0.041271           0.516667            0.169855       155905         256          10       0.008413         0.496331            4.0    0.592198
t2_objective coin_multitimeframe_structure ITransformerLike  correlation_huber seasonal_diff16 window_standard       16    42    0.006270        1.215498       0.789316    0.053227  -0.318012            0.505000       0.506667                0.4625                       0.465        0.301705         0.003973           0.007234         0.122083 6.526246e+05            533893.75       1.222387              True        0.005158            0.007943           0.767246        0.021574           0.501667            0.152330       155905         256          10       0.007603         0.488109            4.3    0.580962
t2_objective coin_multitimeframe_structure ITransformerLike       regime_huber seasonal_diff16 window_standard       16    42    0.006477        1.255778       0.815472    0.055198  -0.383110            0.514583       0.505000                0.4625                       0.465        0.378237         0.004449           0.007234         0.131250 6.743764e+05            533893.75       1.263128              True        0.005158            0.007943           0.778196        0.032480           0.525000            0.190368       155905         256          10       0.009173         0.642679            4.1    0.578651
t2_objective coin_multitimeframe_structure ITransformerLike     variance_huber seasonal_diff16 window_standard       16    42    0.006435        1.247470       0.810078    0.033873  -0.378946            0.507917       0.523333                0.4625                       0.465        0.372164         0.004413           0.007234         0.120833 6.708131e+05            533893.75       1.256454              True        0.005158            0.007943           0.791047       -0.008215           0.495000            0.197833       155905         256          10       0.008004         2.793224            4.1    0.576198
t2_objective coin_multitimeframe_structure ITransformerLike balanced_composite seasonal_diff16 window_standard       16    42    0.006442        1.248971       0.811052    0.044347  -0.384901            0.506250       0.511667                0.4625                       0.465        0.371637         0.004410           0.007234         0.108750 6.701467e+05            533893.75       1.255206              True        0.005158            0.007943           0.778097        0.040708           0.501667            0.189156       155905         256          10       0.008831         2.972314            3.5    0.574908
t2_objective coin_multitimeframe_structure     PatchTSTLike     variance_huber seasonal_diff16 window_standard       16    42    0.006482        1.256753       0.816106    0.028579  -0.389060            0.504167       0.516667                0.4625                       0.465        0.403761         0.004597           0.007234         0.120417 6.773320e+05            533893.75       1.268664              True        0.005158            0.007943           0.785170       -0.030673           0.485000            0.177817       162049         256          10       0.007953         0.888029            4.1    0.563635
t2_objective coin_multitimeframe_structure     PatchTSTLike balanced_composite seasonal_diff16 window_standard       16    42    0.006010        1.165227       0.756671    0.012832  -0.239315            0.516667       0.515000                0.4625                       0.465        0.252189         0.003633           0.007234         0.155417 6.254683e+05            533893.75       1.171522              True        0.005158            0.007943           0.747377       -0.053689           0.446667            0.130392       162049         256          10       0.008423         2.005509            3.6    0.552165
t2_objective coin_multitimeframe_structure     PatchTSTLike  correlation_huber seasonal_diff16 window_standard       16    42    0.005685        1.102088       0.715670    0.001629  -0.148789            0.510833       0.490000                0.4625                       0.465        0.142423         0.002730           0.007234         0.216667 5.898076e+05            533893.75       1.104728              True        0.005158            0.007943           0.714787       -0.034815           0.468333            0.077960       162049         256          10       0.007052         0.217731            4.1    0.520062
t2_objective coin_multitimeframe_structure     PatchTSTLike              huber seasonal_diff16 window_standard       16    42    0.005675        1.100214       0.714453   -0.010947  -0.147432            0.520417       0.501667                0.4625                       0.465        0.123523         0.002542           0.007234         0.209167 5.893316e+05            533893.75       1.103837              True        0.005158            0.007943           0.713968       -0.054372           0.460000            0.070439       162049         256          10       0.006397         0.209182            3.9    0.519274
t2_objective coin_multitimeframe_structure     PatchTSTLike         tail_huber seasonal_diff16 window_standard       16    42    0.005613        1.088134       0.706608    0.007905  -0.130715            0.509167       0.481667                0.4625                       0.465        0.119978         0.002506           0.007234         0.207500 5.823844e+05            533893.75       1.090825              True        0.005158            0.007943           0.714116       -0.051296           0.441667            0.064554       162049         256          10       0.008327         0.264144            4.1    0.518911
t2_objective coin_multitimeframe_structure     PatchTSTLike       regime_huber seasonal_diff16 window_standard       16    42    0.005627        1.090851       0.708373    0.005209  -0.131624            0.505417       0.483333                0.4625                       0.465        0.125283         0.002560           0.007234         0.232500 5.839784e+05            533893.75       1.093810              True        0.005158            0.007943           0.713745       -0.060149           0.470000            0.068387       162049         256          10       0.008326         0.270764            4.2    0.517705
t2_objective coin_multitimeframe_structure     PatchTSTLike  directional_huber seasonal_diff16 window_standard       16    42    0.005658        1.096911       0.712308   -0.009832  -0.142594            0.515000       0.496667                0.4625                       0.465        0.135114         0.002659           0.007234         0.224583 5.861850e+05            533893.75       1.097943              True        0.005158            0.007943           0.710756       -0.060092           0.461667            0.072416       162049         256          10       0.007655         0.226595            4.1    0.515604
t2_objective coin_multitimeframe_structure    ModernTCNLike     variance_huber seasonal_diff16 window_standard       16    42    0.005604        1.086433       0.705504   -0.014436  -0.127709            0.507083       0.490000                0.4625                       0.465        0.117763         0.002482           0.007234         0.277917 5.817945e+05            533893.75       1.089720              True        0.005158            0.007943           0.704568       -0.015473           0.475000            0.069158        12577         256           5       0.006295         0.136621            1.3    0.505014
t2_objective coin_multitimeframe_structure    ModernTCNLike   anti_collapse_v2 seasonal_diff16 window_standard       16    42    0.005604        1.086438       0.705508   -0.014437  -0.127713            0.507083       0.490000                0.4625                       0.465        0.117765         0.002482           0.007234         0.277917 5.817974e+05            533893.75       1.089725              True        0.005158            0.007943           0.704569       -0.015472           0.475000            0.069160        12577         256           5       0.006295         2.301000            1.3    0.505012
t2_objective coin_multitimeframe_structure    ModernTCNLike balanced_composite seasonal_diff16 window_standard       16    42    0.005604        1.086437       0.705507   -0.014438  -0.127714            0.507083       0.490000                0.4625                       0.465        0.117765         0.002482           0.007234         0.277917 5.817969e+05            533893.75       1.089724              True        0.005158            0.007943           0.704568       -0.015472           0.475000            0.069159        12577         256           5       0.006295         1.082607            1.5    0.505011
t2_objective coin_multitimeframe_structure    ModernTCNLike       regime_huber seasonal_diff16 window_standard       16    42    0.005604        1.086439       0.705508   -0.014441  -0.127715            0.507083       0.490000                0.4625                       0.465        0.117765         0.002482           0.007234         0.277917 5.817979e+05            533893.75       1.089726              True        0.005158            0.007943           0.704568       -0.015473           0.475000            0.069159        12577         256           5       0.006295         0.161214            1.3    0.505008
t2_objective coin_multitimeframe_structure    ModernTCNLike         tail_huber seasonal_diff16 window_standard       16    42    0.005604        1.086439       0.705508   -0.014446  -0.127717            0.507083       0.490000                0.4625                       0.465        0.117763         0.002482           0.007234         0.277917 5.817976e+05            533893.75       1.089725              True        0.005158            0.007943           0.704567       -0.015471           0.475000            0.069158        12577         256           5       0.006295         0.155052            1.3    0.505003
t2_objective coin_multitimeframe_structure    ModernTCNLike  directional_huber seasonal_diff16 window_standard       16    42    0.005604        1.086440       0.705509   -0.014447  -0.127719            0.507083       0.490000                0.4625                       0.465        0.117764         0.002482           0.007234         0.277917 5.817984e+05            533893.75       1.089727              True        0.005158            0.007943           0.704567       -0.015470           0.475000            0.069158        12577         256           5       0.006295         0.140897            1.3    0.505002
t2_objective coin_multitimeframe_structure ITransformerLike   anti_collapse_v2 seasonal_diff16 window_standard       16    42    0.022648        4.390827       2.851300    0.026559 -14.773087            0.511250       0.508333                0.4625                       0.465       14.362652         0.027415           0.007234         0.019167 2.388827e+06            533893.75       4.474349              True        0.005158            0.007943           2.292234        0.021595           0.493333            7.160982       155905         256           5       0.022008       164.671869            1.0    0.349762
t2_objective coin_multitimeframe_structure     PatchTSTLike   anti_collapse_v2 seasonal_diff16 window_standard       16    42    0.025289        4.902856       3.183800   -0.054551 -17.458369            0.505000       0.513333                0.4625                       0.465        8.058540         0.020535           0.007234         0.021667 2.610751e+06            533893.75       4.890019              True        0.005158            0.007943           2.499435       -0.011715           0.495000            4.147954       162049         256           5       0.024282       103.340203            1.0    0.240402
t2_objective coin_multitimeframe_structure    ModernTCNLike              huber seasonal_diff16 window_standard       16    42    0.005231        1.014080       0.658520    0.033668  -0.018454            0.520417       0.535000                0.4625                       0.465        0.022926         0.001095           0.007234         0.531667 5.419015e+05            533893.75       1.014999             False        0.005158            0.007943           0.668386       -0.003750           0.486667            0.012668        12577         256          10       0.005925         0.092700            2.4   -0.397184
t2_objective coin_multitimeframe_structure    ModernTCNLike  correlation_huber seasonal_diff16 window_standard       16    42    0.005268        1.021273       0.663191    0.030510  -0.030254            0.513333       0.496667                0.4625                       0.465        0.042785         0.001496           0.007234         0.413750 5.459726e+05            533893.75       1.022624             False        0.005158            0.007943           0.679334       -0.047988           0.476667            0.022670        12577         256           6       0.006246         0.136174            1.6   -0.439142
```

## objective별 평균

```
         objective  trend_corr  large_move_da  mase_momentum  mae_zero_ratio  variance_ratio  direction_accuracy
  anti_collapse_v2     -0.0141         0.5039         2.2469          3.4600          7.5130              0.5078
balanced_composite      0.0142         0.5056         0.7577          1.1669          0.2472              0.5100
 correlation_huber      0.0285         0.4978         0.7227          1.1130          0.1623              0.5097
 directional_huber      0.0105         0.5011         0.7404          1.1402          0.1953              0.5103
             huber      0.0313         0.5200         0.7184          1.1063          0.1674              0.5207
      regime_huber      0.0153         0.4928         0.7431          1.1444          0.2071              0.5090
        tail_huber      0.0207         0.4978         0.7320          1.1273          0.1839              0.5110
    variance_huber      0.0160         0.5100         0.7772          1.1969          0.2979              0.5064
```
