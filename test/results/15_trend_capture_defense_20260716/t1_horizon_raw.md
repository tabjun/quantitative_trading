# 15번 raw 결과 — t1_horizon

- 실행 시각: 2026-07-16 02:26:31 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t1_horizon
- 케이스 수: 24 (실패 0)

## 실행 인자

```json
{
  "suite": "t1_horizon",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "PatchTSTLike,DLinearLike,NLinearLike,TCN,ModernTCNLike,ITransformerLike",
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
  "memory_available_gb": 25.925498962402344,
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
t1_horizon coin_multitimeframe_structure ITransformerLike balanced_composite seasonal_diff16 window_standard       64    42    0.011337        1.110208       0.794227    0.039934  -0.188841            0.513333       0.533333              0.527917                    0.526667        0.220831         0.006347           0.013505         0.169583 1.181173e+06         1.058759e+06       1.115620              True        0.010211            0.014274           0.788801        0.009864           0.521667            0.106706       155905         256          10       0.017426         9.190520            4.0    0.593845
t1_horizon coin_multitimeframe_structure ITransformerLike balanced_composite seasonal_diff16 window_standard       16    42    0.006442        1.248971       0.811052    0.044347  -0.384901            0.506250       0.511667              0.462500                    0.465000        0.371637         0.004410           0.007234         0.108750 6.701467e+05         5.338938e+05       1.255206              True        0.005158            0.007943           0.778097        0.040708           0.501667            0.189156       155905         256          10       0.008831         2.972314            3.6    0.574908
t1_horizon coin_multitimeframe_structure    ModernTCNLike balanced_composite seasonal_diff16 window_standard        4    42    0.003015        1.123464       0.756532   -0.005068  -0.173358            0.500417       0.536667              0.478333                    0.470000        0.148803         0.001480           0.003835         0.210000 3.126933e+05         2.773617e+05       1.127385              True        0.002683            0.003985           0.749091       -0.043802           0.493333            0.082273        12577         256           6       0.003351         0.662257            1.8    0.555946
t1_horizon coin_multitimeframe_structure     PatchTSTLike balanced_composite seasonal_diff16 window_standard       16    42    0.006010        1.165227       0.756671    0.012832  -0.239315            0.516667       0.515000              0.462500                    0.465000        0.252189         0.003633           0.007234         0.155417 6.254683e+05         5.338938e+05       1.171522              True        0.005158            0.007943           0.747377       -0.053689           0.446667            0.130392       162049         256          10       0.008423         2.005509            4.4    0.552165
t1_horizon coin_multitimeframe_structure      DLinearLike balanced_composite seasonal_diff16 window_standard        1    42    0.001336        1.057963       0.687260    0.001408  -0.063213            0.510833       0.511667              0.466250                    0.460000        0.062472         0.000478           0.001911         0.300833 1.382065e+05         1.303838e+05       1.059998              True        0.001263            0.001944           0.723387       -0.001240           0.518333            0.031539          147         256          10       0.001781         0.863768            2.7    0.544349
t1_horizon coin_multitimeframe_structure    ModernTCNLike balanced_composite seasonal_diff16 window_standard       16    42    0.005604        1.086439       0.705508   -0.014445  -0.127716            0.507083       0.490000              0.462500                    0.465000        0.117763         0.002482           0.007234         0.277917 5.817973e+05         5.338938e+05       1.089725              True        0.005158            0.007943           0.704568       -0.015471           0.475000            0.069158        12577         256           5       0.006295         1.160577            1.5    0.505004
t1_horizon coin_multitimeframe_structure    ModernTCNLike balanced_composite seasonal_diff16 window_standard        1    42    0.001605        1.271071       0.825697   -0.027115  -0.446326            0.504583       0.510000              0.466250                    0.460000        0.411422         0.001226           0.001911         0.166667 1.668331e+05         1.303838e+05       1.279555              True        0.001263            0.001944           0.810138        0.007665           0.508333            0.219835        12577         256           7       0.001913         0.925173            0.8    0.500316
t1_horizon coin_multitimeframe_structure     PatchTSTLike balanced_composite seasonal_diff16 window_standard        4    42    0.003505        1.306168       0.879563    0.011219  -0.464392            0.511667       0.476667              0.478333                    0.470000        0.431352         0.002519           0.003835         0.120833 3.651872e+05         2.773617e+05       1.316646              True        0.002683            0.003985           0.852092        0.010659           0.485000            0.244854       162049         256          10       0.004504         1.842793            4.4    0.499930
t1_horizon coin_multitimeframe_structure ITransformerLike balanced_composite seasonal_diff16 window_standard        4    42    0.004605        1.715953       1.155509   -0.001884  -1.411386            0.507917       0.513333              0.478333                    0.470000        1.388097         0.004519           0.003835         0.070000 4.810704e+05         2.773617e+05       1.734452              True        0.002683            0.003985           1.055085       -0.001598           0.498333            0.734057       155905         256          10       0.005736         1.465676            4.4    0.495898
t1_horizon coin_multitimeframe_structure      DLinearLike balanced_composite seasonal_diff16 window_standard       64    42    0.010716        1.049465       0.750772   -0.035387  -0.082471            0.481667       0.488333              0.527917                    0.526667        0.062913         0.003387           0.013505         0.329583 1.112121e+06         1.058759e+06       1.050400              True        0.010211            0.014274           0.736862        0.004872           0.510000            0.029015          147         256          10       0.016140         1.097429            2.7    0.477869
t1_horizon coin_multitimeframe_structure      NLinearLike balanced_composite seasonal_diff16 window_standard       16    42    0.010385        2.013303       1.307391   -0.014160  -2.418301            0.498750       0.521667              0.462500                    0.465000        2.369654         0.011136           0.007234         0.055000 1.086290e+06         5.338938e+05       2.034656              True        0.005158            0.007943           1.175766       -0.012995           0.493333            1.297690           82         256          10       0.013893         2.927931            2.6    0.476767
t1_horizon coin_multitimeframe_structure     PatchTSTLike balanced_composite seasonal_diff16 window_standard        1    42    0.002407        1.905710       1.237963   -0.004872  -1.611263            0.499583       0.470000              0.466250                    0.460000        1.592787         0.002412           0.001911         0.060417 2.516851e+05         1.303838e+05       1.930341              True        0.001263            0.001944           1.084973       -0.011202           0.506667            0.817741       162049         256          10       0.002962         1.893475            4.0    0.441332
t1_horizon coin_multitimeframe_structure      NLinearLike balanced_composite seasonal_diff16 window_standard       64    42    0.016247        1.591062       1.138223   -0.021917  -1.317133            0.481250       0.475000              0.527917                    0.526667        1.267695         0.015206           0.013505         0.071667 1.696208e+06         1.058759e+06       1.602071              True        0.010211            0.014274           1.006887        0.016066           0.521667            0.623706           82         256           9       0.022720         5.228520            2.4    0.439261
t1_horizon coin_multitimeframe_structure      NLinearLike balanced_composite seasonal_diff16 window_standard        4    42    0.009662        3.600735       2.424706    0.007349  -9.528752            0.505000       0.526667              0.478333                    0.470000        9.569575         0.011865           0.003835         0.032917 1.013767e+06         2.773617e+05       3.655037              True        0.002683            0.003985           2.106556       -0.006463           0.491667            5.409268           82         256          10       0.012263         0.889541            2.6    0.391545
t1_horizon coin_multitimeframe_structure ITransformerLike balanced_composite seasonal_diff16 window_standard        1    42    0.003651        2.890696       1.877817   -0.017951  -5.016875            0.503333       0.485000              0.466250                    0.460000        4.900220         0.004231           0.001911         0.034167 3.832779e+05         1.303838e+05       2.939614              True        0.001263            0.001944           1.495998        0.009000           0.491667            2.322293       155905         256          10       0.004273         2.965349            3.4    0.379268
t1_horizon coin_multitimeframe_structure     PatchTSTLike balanced_composite seasonal_diff16 window_standard       64    42    0.013436        1.315842       0.941334   -0.078215  -0.602841            0.481667       0.445000              0.527917                    0.526667        0.474604         0.009304           0.013505         0.121667 1.411314e+06         1.058759e+06       1.332988              True        0.010211            0.014274           0.848148        0.045850           0.508333            0.235973       162049         256          10       0.018881        15.627985            4.1    0.372652
t1_horizon coin_multitimeframe_structure      DLinearLike balanced_composite seasonal_diff16 window_standard        4    42    0.002730        1.017514       0.685186    0.031380  -0.021784            0.517917       0.543333              0.478333                    0.470000        0.033192         0.000699           0.003835         0.425000 2.822263e+05         2.773617e+05       1.017539             False        0.002683            0.003985           0.694341       -0.008927           0.511667            0.016699          147         256          10       0.003538         1.897107            2.7   -0.393805
t1_horizon coin_multitimeframe_structure              TCN balanced_composite seasonal_diff16 window_standard       16    42    0.005378        1.042680       0.677092    0.015169  -0.048642            0.508750       0.520000              0.462500                    0.465000        0.007378         0.000621           0.007234         0.057500 5.570873e+05         5.338938e+05       1.043442             False        0.005158            0.007943           0.680176        0.018913           0.513333            0.003774        88225         256           5       0.006069         1.110218            1.6   -0.432540
t1_horizon coin_multitimeframe_structure              TCN balanced_composite seasonal_diff16 window_standard       64    42    0.010199        0.998795       0.714524    0.044149  -0.004276            0.509583       0.486667              0.527917                    0.526667        0.002474         0.000672           0.013505         0.894583 1.058164e+06         1.058759e+06       0.999437             False        0.010211            0.014274           0.719266        0.021881           0.435000            0.001141        88225         256           5       0.012534         5.322530            1.6   -0.440637
t1_horizon coin_multitimeframe_structure              TCN balanced_composite seasonal_diff16 window_standard        4    42    0.002986        1.112627       0.749234    0.002107  -0.135539            0.500417       0.513333              0.478333                    0.470000        0.016284         0.000489           0.003835         0.015417 3.097601e+05         2.773617e+05       1.116809             False        0.002683            0.003985           0.739051        0.010825           0.516667            0.008708        88225         256           5       0.003051         0.487723            1.6   -0.459483
t1_horizon coin_multitimeframe_structure              TCN balanced_composite seasonal_diff16 window_standard        1    42    0.001450        1.147881       0.745672    0.011562  -0.156649            0.485000       0.480000              0.466250                    0.460000        0.033953         0.000352           0.001911         0.086667 1.503119e+05         1.303838e+05       1.152842             False        0.001263            0.001944           0.741594        0.013993           0.530000            0.017619        88225         256           6       0.001590         0.469894            1.0   -0.483005
t1_horizon coin_multitimeframe_structure      DLinearLike balanced_composite seasonal_diff16 window_standard       16    42    0.005375        1.042122       0.676730   -0.013660  -0.056547            0.478333       0.461667              0.462500                    0.465000        0.041112         0.001467           0.007234         0.371667 5.571326e+05         5.338938e+05       1.043527             False        0.005158            0.007943           0.675439        0.006406           0.486667            0.021808          147         256          10       0.007517         1.250944            2.7   -0.519666
t1_horizon coin_multitimeframe_structure    ModernTCNLike balanced_composite seasonal_diff16 window_standard       64    42    0.010503        1.028546       0.735807   -0.027666  -0.062424            0.507917       0.446667              0.527917                    0.526667        0.034169         0.002496           0.013505         0.387500 1.091022e+06         1.058759e+06       1.030473             False        0.010211            0.014274           0.744940       -0.052745           0.468333            0.016532        12577         256           5       0.013015         0.898653            1.5   -0.554580
t1_horizon coin_multitimeframe_structure      NLinearLike balanced_composite seasonal_diff16 window_standard        1    42    0.008662        6.858594       4.455393   -0.042368 -33.190020            0.487917       0.478333              0.466250                    0.460000       32.659734         0.010924           0.001911         0.015000 9.092529e+05         1.303838e+05       6.973667             False        0.001263            0.001944           3.437163       -0.031295           0.471667           16.458511           82         256          10       0.010679         1.549426            2.5   -0.909574
```

## horizon별 평균

```
 horizon  trend_corr  large_move_da  mase_momentum  mae_zero_ratio  variance_ratio  direction_accuracy
       1     -0.0132         0.4892         1.6383          2.5220          6.6101              0.4985
       4      0.0075         0.5183         1.1085          1.6461          1.9312              0.5072
      16      0.0050         0.5033         0.8224          1.2665          0.5266              0.5026
      64     -0.0132         0.4792         0.8458          1.1823          0.3438              0.4959
```
