# 15번 raw 결과 — t7_amplitude

- 실행 시각: 2026-07-16 12:48:56 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t7_amplitude
- 케이스 수: 24 (실패 0)

## 실행 인자

```json
{
  "suite": "t7_amplitude",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "ITransformerLike,Linear",
  "objectives": "huber,tail_huber,variance_huber,regime_huber",
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
  "max_windows": 24000,
  "stride": 2,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "cross_tickers": "",
  "ensemble_seeds": "42,7,123,2026",
  "tickers": "KRW-BTC,KRW-ETH,KRW-XRP,KRW-SOL",
  "ticker_tables": "",
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
  "memory_available_gb": 25.56057357788086,
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

## 이 suite가 보는 것 — 진폭 교정 — tail 가중 + 분산 보존

- **고정**: 우승 h·정규화·변수셋을 고정한다.
- **변화**: tail 가중 손실과 분산 보존 objective, 전처리를 바꿔 진폭 과소예측을 교정한다.
- **질문**: variance_ratio를 0.36에서 1 근처로 끌어올리면서 trend_corr를 유지할 수 있는가?
- **읽는 법**: variance_ratio가 1에 가까워지고 large_move_da가 오르면 큰 변동 포착 개선.

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
       suite                   feature_set            model      objective   preprocessing   normalization  horizon  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr        r2  direction_accuracy  large_move_da  momentum_da_baseline  momentum_large_da_baseline  variance_ratio  pred_return_std  actual_return_std  near_zero_share       mae_krw  persistence_mae_krw  copy_risk_krw  healthy_variance  naive_mae_zero  naive_mae_momentum  val_mase_momentum  val_trend_corr  val_large_move_da  val_variance_ratio  param_count  batch_size  epochs_run  best_val_loss  final_grad_norm  train_seconds  amp_gap
t7_amplitude coin_multitimeframe_structure ITransformerLike variance_huber seasonal_diff16 window_standard       16    42    0.006842        1.299771       0.844245    0.072895 -0.508463            0.517222       0.512222              0.455833                    0.474444        0.548047         0.005448           0.007359         0.106667 698656.731044        532931.666667       1.310969              True        0.005264            0.008105           0.823802        0.052219           0.542222            0.426484       155905         256          10       0.007642         3.749140            6.3 0.451953
t7_amplitude coin_multitimeframe_structure ITransformerLike variance_huber            none window_standard       16    42    0.006046        1.148535       0.746012    0.054526 -0.213301            0.520556       0.510000              0.455833                    0.474444        0.268128         0.003811           0.007359         0.166667 616093.097313        532931.666667       1.156045              True        0.005264            0.008105           0.759097       -0.029821           0.507778            0.245897       155905         256          10       0.007030         3.765369            6.1 0.731872
t7_amplitude coin_multitimeframe_structure           Linear variance_huber            none window_standard       16    42    0.005949        1.130100       0.734038    0.053115 -0.183812            0.503056       0.515556              0.455833                    0.474444        0.222712         0.003473           0.007359         0.181667 606250.322745        532931.666667       1.137576              True        0.005264            0.008105           0.719342       -0.002771           0.491111            0.152272        98497         256          10       0.006567         0.242618            3.2 0.777288
t7_amplitude coin_multitimeframe_structure           Linear     tail_huber            none window_standard       16    42    0.005902        1.121158       0.728230    0.037974 -0.173473            0.513889       0.517778              0.455833                    0.474444        0.206153         0.003341           0.007359         0.188056 600234.019571        532931.666667       1.126287              True        0.005264            0.008105           0.715245        0.000604           0.493333            0.139849        98497         256          10       0.007617         0.279421            2.2 0.793847
t7_amplitude coin_multitimeframe_structure           Linear variance_huber      winsor_025 window_standard       16    42    0.005898        1.120421       0.727751    0.024968 -0.175225            0.511389       0.537778              0.455833                    0.474444        0.187407         0.003186           0.007359         0.197778 600130.175198        532931.666667       1.126092              True        0.005264            0.008105           0.710250       -0.013082           0.478889            0.128800        98497         256          10       0.006533         0.196550            3.2 0.812593
t7_amplitude coin_multitimeframe_structure           Linear     tail_huber      winsor_025 window_standard       16    42    0.005944        1.129165       0.733431    0.012479 -0.179164            0.496111       0.515556              0.455833                    0.474444        0.186336         0.003177           0.007359         0.196944 604707.207206        532931.666667       1.134681              True        0.005264            0.008105           0.708314       -0.010066           0.488889            0.131293        98497         256          10       0.007630         0.260770            1.9 0.813664
t7_amplitude coin_multitimeframe_structure           Linear   regime_huber      winsor_025 window_standard       16    42    0.005986        1.137099       0.738584    0.000383 -0.190054            0.492222       0.511111              0.455833                    0.474444        0.185410         0.003169           0.007359         0.199722 609303.179548        532931.666667       1.143305              True        0.005264            0.008105           0.707402       -0.010563           0.477778            0.125387        98497         256          10       0.007507         0.268053            1.4 0.814590
t7_amplitude coin_multitimeframe_structure           Linear          huber      winsor_025 window_standard       16    42    0.005974        1.134839       0.737116    0.008947 -0.192110            0.498333       0.498889              0.455833                    0.474444        0.184779         0.003163           0.007359         0.201111 608015.262210        532931.666667       1.140888              True        0.005264            0.008105           0.713750       -0.019590           0.478889            0.125031        98497         256          10       0.005856         0.233323            3.0 0.815221
t7_amplitude coin_multitimeframe_structure           Linear variance_huber seasonal_diff16 window_standard       16    42    0.005820        1.105663       0.718165    0.055494 -0.137172            0.521944       0.536667              0.455833                    0.474444        0.182495         0.003144           0.007359         0.190000 590770.196088        532931.666667       1.108529              True        0.005264            0.008105           0.721288       -0.042519           0.464444            0.132884        98497         256          10       0.006669         0.243821            3.2 0.817505
t7_amplitude coin_multitimeframe_structure           Linear          huber            none window_standard       16    42    0.005845        1.110311       0.721185    0.027190 -0.154926            0.510833       0.527778              0.455833                    0.474444        0.174478         0.003074           0.007359         0.205000 594618.824993        532931.666667       1.115751              True        0.005264            0.008105           0.707688        0.006428           0.505556            0.121879        98497         256          10       0.005783         0.240818            3.0 0.825522
t7_amplitude coin_multitimeframe_structure ITransformerLike   regime_huber seasonal_diff16 window_standard       16    42    0.005758        1.093897       0.710523    0.049356 -0.136708            0.521667       0.507778              0.455833                    0.474444        0.171898         0.003051           0.007359         0.202778 585124.791056        532931.666667       1.097936              True        0.005264            0.008105           0.706483        0.026682           0.510000            0.126416       155905         256          10       0.007509         0.577681            6.1 0.828102
t7_amplitude coin_multitimeframe_structure           Linear   regime_huber            none window_standard       16    42    0.005861        1.113318       0.723137    0.009254 -0.166807            0.505833       0.514444              0.455833                    0.474444        0.170962         0.003043           0.007359         0.191944 595849.998559        532931.666667       1.118061              True        0.005264            0.008105           0.710067       -0.011882           0.493333            0.116801        98497         256          10       0.007540         0.277228            2.5 0.829038
t7_amplitude coin_multitimeframe_structure ITransformerLike          huber seasonal_diff16 window_standard       16    42    0.005784        1.098776       0.713692    0.040986 -0.145253            0.514722       0.492222              0.455833                    0.474444        0.170782         0.003041           0.007359         0.212222 587888.914736        532931.666667       1.103123              True        0.005264            0.008105           0.706266        0.019385           0.515556            0.126039       155905         256          10       0.005761         0.456660            6.6 0.829218
t7_amplitude coin_multitimeframe_structure ITransformerLike     tail_huber seasonal_diff16 window_standard       16    42    0.005722        1.086967       0.706021    0.060541 -0.115589            0.511944       0.520000              0.455833                    0.474444        0.161919         0.002961           0.007359         0.207778 581368.931647        532931.666667       1.090888              True        0.005264            0.008105           0.707643        0.016921           0.516667            0.120328       155905         256          10       0.007503         0.573512            6.1 0.838081
t7_amplitude coin_multitimeframe_structure ITransformerLike     tail_huber            none window_standard       16    42    0.005883        1.117545       0.725883    0.011024 -0.169429            0.507500       0.482222              0.455833                    0.474444        0.152427         0.002873           0.007359         0.207778 599236.589451        532931.666667       1.124415              True        0.005264            0.008105           0.710953       -0.009919           0.508889            0.112534       155905         256          10       0.007497         0.718477            6.1 0.847573
t7_amplitude coin_multitimeframe_structure           Linear     tail_huber seasonal_diff16 window_standard       16    42    0.005768        1.095713       0.711703    0.005951 -0.134418            0.503611       0.506667              0.455833                    0.474444        0.129434         0.002648           0.007359         0.224167 585751.714391        532931.666667       1.099112              True        0.005264            0.008105           0.699374       -0.019190           0.485556            0.094094        98497         256          10       0.007440         0.272767            3.0 0.870566
t7_amplitude coin_multitimeframe_structure           Linear   regime_huber seasonal_diff16 window_standard       16    42    0.005754        1.092972       0.709922   -0.008427 -0.137324            0.510833       0.502222              0.455833                    0.474444        0.128256         0.002636           0.007359         0.233056 584307.008473        532931.666667       1.096401              True        0.005264            0.008105           0.699397       -0.027154           0.490000            0.089608        98497         256          10       0.007460         0.274723            2.1 0.871744
t7_amplitude coin_multitimeframe_structure ITransformerLike   regime_huber      winsor_025 window_standard       16    42    0.005773        1.096741       0.712370    0.026808 -0.137844            0.515000       0.503333              0.455833                    0.474444        0.121755         0.002568           0.007359         0.223889 587626.539407        532931.666667       1.102630              True        0.005264            0.008105           0.706496       -0.016193           0.511111            0.094892       155905         256          10       0.007428         0.749315            6.1 0.878245
t7_amplitude coin_multitimeframe_structure ITransformerLike     tail_huber      winsor_025 window_standard       16    42    0.005727        1.087848       0.706594    0.024555 -0.125521            0.520278       0.510000              0.455833                    0.474444        0.121089         0.002561           0.007359         0.230000 582915.932013        532931.666667       1.093791              True        0.005264            0.008105           0.703277       -0.018374           0.494444            0.093152       155905         256          10       0.007399         0.755968            6.0 0.878911
t7_amplitude coin_multitimeframe_structure ITransformerLike   regime_huber            none window_standard       16    42    0.005825        1.106581       0.718762    0.021618 -0.147060            0.505000       0.496667              0.455833                    0.474444        0.117277         0.002520           0.007359         0.221944 593568.661612        532931.666667       1.113780              True        0.005264            0.008105           0.697163       -0.007802           0.517778            0.081531       155905         256          10       0.007281         0.780315            6.2 0.882723
t7_amplitude coin_multitimeframe_structure ITransformerLike          huber            none window_standard       16    42    0.005784        1.098666       0.713621    0.018702 -0.136744            0.509722       0.501111              0.455833                    0.474444        0.113626         0.002481           0.007359         0.232500 589122.236150        532931.666667       1.105437              True        0.005264            0.008105           0.698845       -0.009840           0.520000            0.080190       155905         256          10       0.005643         0.669796            6.0 0.886374
t7_amplitude coin_multitimeframe_structure ITransformerLike          huber      winsor_025 window_standard       16    42    0.005758        1.093884       0.710514    0.016961 -0.134058            0.511667       0.494444              0.455833                    0.474444        0.112794         0.002472           0.007359         0.230833 586142.443912        532931.666667       1.099845              True        0.005264            0.008105           0.699364       -0.010730           0.516667            0.085001       155905         256          10       0.005650         0.634900            5.8 0.887206
t7_amplitude coin_multitimeframe_structure ITransformerLike variance_huber      winsor_025 window_standard       16    42    0.005908        1.122365       0.729014    0.049987 -0.164303            0.498611       0.488889              0.455833                    0.474444        0.109318         0.002433           0.007359         0.194167 601839.727231        532931.666667       1.129300              True        0.005264            0.008105           0.718477       -0.029684           0.498889            0.093173       155905         256          10       0.006474         0.542513            6.2 0.890682
t7_amplitude coin_multitimeframe_structure           Linear          huber seasonal_diff16 window_standard       16    42    0.005693        1.081499       0.702470    0.001821 -0.112599            0.503333       0.497778              0.455833                    0.474444        0.106158         0.002398           0.007359         0.243333 577921.173888        532931.666667       1.084419              True        0.005264            0.008105           0.688418       -0.013384           0.475556            0.074534        98497         256          10       0.005658         0.215451            3.0 0.893842
```

## 진폭 교정 순위 (variance_ratio가 1에 가까울수록 상단)

```
           model      objective   preprocessing  variance_ratio  trend_corr  large_move_da  amp_gap
ITransformerLike variance_huber seasonal_diff16          0.5480      0.0729         0.5122   0.4520
ITransformerLike variance_huber            none          0.2681      0.0545         0.5100   0.7319
          Linear variance_huber            none          0.2227      0.0531         0.5156   0.7773
          Linear     tail_huber            none          0.2062      0.0380         0.5178   0.7938
          Linear variance_huber      winsor_025          0.1874      0.0250         0.5378   0.8126
          Linear     tail_huber      winsor_025          0.1863      0.0125         0.5156   0.8137
          Linear   regime_huber      winsor_025          0.1854      0.0004         0.5111   0.8146
          Linear          huber      winsor_025          0.1848      0.0089         0.4989   0.8152
          Linear variance_huber seasonal_diff16          0.1825      0.0555         0.5367   0.8175
          Linear          huber            none          0.1745      0.0272         0.5278   0.8255
ITransformerLike   regime_huber seasonal_diff16          0.1719      0.0494         0.5078   0.8281
          Linear   regime_huber            none          0.1710      0.0093         0.5144   0.8290
```
