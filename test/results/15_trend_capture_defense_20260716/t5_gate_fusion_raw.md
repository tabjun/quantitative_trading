# 15번 raw 결과 — t5_gate_fusion

- 실행 시각: 2026-07-16 02:38:55 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t5_gate_fusion
- 케이스 수: 36 (실패 0)

## 실행 인자

```json
{
  "suite": "t5_gate_fusion",
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
  "feature_sets": "coin_multitimeframe_structure,mtf_plus_momentum,mtf_trend",
  "seeds": "42,7,123",
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
  "memory_available_gb": 25.852210998535156,
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
         suite                   feature_set            model objective   preprocessing   normalization  horizon  seed gate_quantile  gate_cutoff  risk_ap  risk_ap_lift  risk_event_rate  trend_corr  large_move_da  mase_momentum  variance_ratio  cumulative_return       mdd  sortino_proxy  active_share  trade_count  turnover  mean_step_pnl  decision_count  blocked_long_share  qualified
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.45     0.463454 0.392425      1.391168         0.282083    0.014940       0.503333       0.782835        0.319569          -0.133669 -0.134182      -0.063546      0.095000           96      96.0      -0.000239             600            0.118333       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.55     0.495906 0.392425      1.391168         0.282083    0.014940       0.503333       0.782835        0.319569          -0.163894 -0.164059      -0.081512      0.110000          110     110.0      -0.000298             600            0.103333       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.65     0.531772 0.392425      1.391168         0.282083    0.014940       0.503333       0.782835        0.319569          -0.180376 -0.180538      -0.095429      0.128333          128     128.0      -0.000332             600            0.085000       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16     7          none          NaN 0.392425      1.391168         0.282083    0.014940       0.503333       0.782835        0.319569          -0.269218 -0.289226      -0.128060      0.213333          208     208.0      -0.000523             600            0.000000       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.45     0.498627 0.381956      1.354053         0.282083    0.071235       0.523333       0.782249        0.355765          -0.122667 -0.156735      -0.080904      0.163333          147     147.0      -0.000218             600            0.245000       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.55     0.534819 0.381956      1.354053         0.282083    0.071235       0.523333       0.782249        0.355765          -0.119354 -0.159455      -0.070027      0.196667          171     171.0      -0.000212             600            0.211667       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.65     0.569346 0.381956      1.354053         0.282083    0.071235       0.523333       0.782249        0.355765          -0.117990 -0.178810      -0.058526      0.240000          197     197.0      -0.000209             600            0.168333       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42          none          NaN 0.381956      1.354053         0.282083    0.071235       0.523333       0.782249        0.355765          -0.272728 -0.297807      -0.131035      0.408333          295     295.0      -0.000531             600            0.000000       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.45     0.418857 0.402726      1.427686         0.282083    0.009655       0.478333       0.817181        0.421580          -0.191355 -0.204084      -0.118000      0.168333          158     158.0      -0.000354             600            0.228333       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.55     0.466842 0.402726      1.427686         0.282083    0.009655       0.478333       0.817181        0.421580          -0.260067 -0.271714      -0.154469      0.206667          192     192.0      -0.000502             600            0.190000       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.65     0.510909 0.402726      1.427686         0.282083    0.009655       0.478333       0.817181        0.421580          -0.287316 -0.297548      -0.175529      0.233333          218     218.0      -0.000565             600            0.163333       True
t5_gate_fusion coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16   123          none          NaN 0.402726      1.427686         0.282083    0.009655       0.478333       0.817181        0.421580          -0.343307 -0.366827      -0.178756      0.396667          314     314.0      -0.000701             600            0.000000       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.45     0.441544 0.391496      1.387875         0.282083    0.025647       0.533333       0.788851        0.339837          -0.119794 -0.134227      -0.068288      0.101667          106     106.0      -0.000213             600            0.123333       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.55     0.482124 0.391496      1.387875         0.282083    0.025647       0.533333       0.788851        0.339837          -0.146653 -0.159370      -0.090015      0.125000          128     128.0      -0.000264             600            0.100000       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.65     0.522515 0.391496      1.387875         0.282083    0.025647       0.533333       0.788851        0.339837          -0.142462 -0.166787      -0.088276      0.153333          150     150.0      -0.000256             600            0.071667       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16     7          none          NaN 0.391496      1.387875         0.282083    0.025647       0.533333       0.788851        0.339837          -0.223263 -0.236791      -0.116617      0.225000          218     218.0      -0.000421             600            0.000000       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.45     0.492336 0.378603      1.342168         0.282083    0.038842       0.501667       0.808366        0.279519          -0.236857 -0.247351      -0.132642      0.248333          185     185.0      -0.000451             600            0.318333       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.55     0.536219 0.378603      1.342168         0.282083    0.038842       0.501667       0.808366        0.279519          -0.263066 -0.273199      -0.152228      0.290000          209     209.0      -0.000509             600            0.276667       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.65     0.581665 0.378603      1.342168         0.282083    0.038842       0.501667       0.808366        0.279519          -0.258983 -0.268631      -0.146239      0.345000          233     233.0      -0.000500             600            0.221667       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16    42          none          NaN 0.378603      1.342168         0.282083    0.038842       0.501667       0.808366        0.279519          -0.375515 -0.380680      -0.192672      0.566667          299     299.0      -0.000785             600            0.000000       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.45     0.547779 0.410282      1.454473         0.282083    0.007825       0.478333       0.762462        0.227102          -0.220342 -0.230501      -0.132181      0.205000          168     168.0      -0.000415             600            0.223333       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.55     0.590389 0.410282      1.454473         0.282083    0.007825       0.478333       0.762462        0.227102          -0.252350 -0.262093      -0.152301      0.230000          184     184.0      -0.000485             600            0.198333       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.65     0.626862 0.410282      1.454473         0.282083    0.007825       0.478333       0.762462        0.227102          -0.253562 -0.261173      -0.142413      0.268333          216     216.0      -0.000487             600            0.160000       True
t5_gate_fusion             mtf_plus_momentum ITransformerLike     huber seasonal_diff16 window_standard       16   123          none          NaN 0.410282      1.454473         0.282083    0.007825       0.478333       0.762462        0.227102          -0.347315 -0.371496      -0.163936      0.428333          312     312.0      -0.000711             600            0.000000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.45     0.479225 0.308684      1.094302         0.282083    0.033529       0.505000       0.711917        0.120793          -0.075351 -0.115326      -0.035632      0.110000          110     110.0      -0.000131             600            0.215000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.55     0.493428 0.308684      1.094302         0.282083    0.033529       0.505000       0.711917        0.120793          -0.039315 -0.107139      -0.019571      0.148333          140     140.0      -0.000067             600            0.176667       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16     7          0.65     0.512102 0.308684      1.094302         0.282083    0.033529       0.505000       0.711917        0.120793          -0.059051 -0.130968      -0.029548      0.180000          164     164.0      -0.000101             600            0.145000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16     7          none          NaN 0.308684      1.094302         0.282083    0.033529       0.505000       0.711917        0.120793          -0.203819 -0.250832      -0.103006      0.325000          262     262.0      -0.000380             600            0.000000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.45     0.564036 0.310942      1.102306         0.282083    0.063103       0.495000       0.759550        0.195535          -0.136058 -0.200619      -0.070928      0.228333          183     183.0      -0.000244             600            0.356667       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.55     0.578499 0.310942      1.102306         0.282083    0.063103       0.495000       0.759550        0.195535          -0.161584 -0.214612      -0.090769      0.273333          215     215.0      -0.000294             600            0.311667       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16    42          0.65     0.595807 0.310942      1.102306         0.282083    0.063103       0.495000       0.759550        0.195535          -0.222427 -0.299043      -0.129179      0.341667          247     247.0      -0.000419             600            0.243333       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16    42          none          NaN 0.310942      1.102306         0.282083    0.063103       0.495000       0.759550        0.195535          -0.370870 -0.411722      -0.207315      0.585000          321     321.0      -0.000772             600            0.000000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.45     0.582676 0.320762      1.137118         0.282083    0.018049       0.516667       0.700862        0.064010          -0.164986 -0.164986      -0.085724      0.148333          136     136.0      -0.000301             600            0.236667       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.55     0.600112 0.320762      1.137118         0.282083    0.018049       0.516667       0.700862        0.064010          -0.231518 -0.231518      -0.121741      0.195000          170     170.0      -0.000439             600            0.190000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16   123          0.65     0.616490 0.320762      1.137118         0.282083    0.018049       0.516667       0.700862        0.064010          -0.293644 -0.293644      -0.160129      0.235000          208     208.0      -0.000579             600            0.150000       True
t5_gate_fusion                     mtf_trend ITransformerLike     huber seasonal_diff16 window_standard       16   123          none          NaN 0.320762      1.137118         0.282083    0.018049       0.516667       0.700862        0.064010          -0.377355 -0.376123      -0.200656      0.385000          296     296.0      -0.000790             600            0.000000       True
```

## gate별 pivot (mdd / cumulative_return / active_share)

```
                                   active_share                         cumulative_return                             mdd                        
gate_quantile                              0.45    0.55    0.65    none              0.45    0.55    0.65    none    0.45    0.55    0.65    none
feature_set                   seed                                                                                                               
coin_multitimeframe_structure 7          0.0950  0.1100  0.1283  0.2133           -0.1337 -0.1639 -0.1804 -0.2692 -0.1342 -0.1641 -0.1805 -0.2892
                              42         0.1633  0.1967  0.2400  0.4083           -0.1227 -0.1194 -0.1180 -0.2727 -0.1567 -0.1595 -0.1788 -0.2978
                              123        0.1683  0.2067  0.2333  0.3967           -0.1914 -0.2601 -0.2873 -0.3433 -0.2041 -0.2717 -0.2975 -0.3668
mtf_plus_momentum             7          0.1017  0.1250  0.1533  0.2250           -0.1198 -0.1467 -0.1425 -0.2233 -0.1342 -0.1594 -0.1668 -0.2368
                              42         0.2483  0.2900  0.3450  0.5667           -0.2369 -0.2631 -0.2590 -0.3755 -0.2474 -0.2732 -0.2686 -0.3807
                              123        0.2050  0.2300  0.2683  0.4283           -0.2203 -0.2524 -0.2536 -0.3473 -0.2305 -0.2621 -0.2612 -0.3715
mtf_trend                     7          0.1100  0.1483  0.1800  0.3250           -0.0754 -0.0393 -0.0591 -0.2038 -0.1153 -0.1071 -0.1310 -0.2508
                              42         0.2283  0.2733  0.3417  0.5850           -0.1361 -0.1616 -0.2224 -0.3709 -0.2006 -0.2146 -0.2990 -0.4117
                              123        0.1483  0.1950  0.2350  0.3850           -0.1650 -0.2315 -0.2936 -0.3774 -0.1650 -0.2315 -0.2936 -0.3761
```
