# 15번 raw 결과 — t6_signal_boost

- 실행 시각: 2026-07-16 12:46:09 (서버, 헤드리스 .py)
- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite t6_signal_boost
- 케이스 수: 15 (실패 0)

## 실행 인자

```json
{
  "suite": "t6_signal_boost",
  "db": null,
  "table": "btc_15m_advance",
  "ticker": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "horizons": "1,4,16,64",
  "models": "ITransformerLike,PatchTSTLike,Linear",
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
  "memory_available_gb": 25.554542541503906,
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

## 이 suite가 보는 것 — 신호 강화 — seed ensemble + 데이터 규모

- **고정**: 우승 h·objective·전처리·정규화·변수셋을 고정한다.
- **변화**: seed(여러 개) 평균 앙상블 여부와 학습 데이터 규모(max_windows/stride)를 바꾼다.
- **질문**: seed 평균과 더 많은 데이터로 trend_corr/large_move_da가 실제로 오르는가?
- **읽는 법**: ensemble_* 행의 trend_corr가 단일 seed 평균보다 높으면 신호 강화 성공. 안 오르면 예측축 한계.

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
          suite                   feature_set            model objective   preprocessing   normalization  horizon  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr        r2  direction_accuracy  large_move_da  momentum_da_baseline  momentum_large_da_baseline  variance_ratio  pred_return_std  actual_return_std  near_zero_share       mae_krw  persistence_mae_krw  copy_risk_krw  healthy_variance  naive_mae_zero  naive_mae_momentum  val_mase_momentum  val_trend_corr  val_large_move_da  val_variance_ratio  param_count  batch_size  epochs_run  best_val_loss  final_grad_norm  train_seconds      member  is_ensemble  single_mean_trend_corr  single_mean_large_move_da  ensemble_corr_gain  ensemble_lda_gain
t6_signal_boost coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    42    0.005784        1.098776       0.713692    0.040986 -0.145253            0.514722       0.492222              0.455833                    0.474444        0.170782         0.003041           0.007359         0.212222 587888.914736        532931.666667       1.103123              True        0.005264            0.008105           0.706266        0.019385           0.515556            0.126039     155905.0       256.0        10.0       0.005761         0.456660            6.0      seed42        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16     7    0.005832        1.107858       0.719591    0.024438 -0.154048            0.510556       0.507778              0.455833                    0.474444        0.147252         0.002824           0.007359         0.213333 592866.322670        532931.666667       1.112462              True        0.005264            0.008105           0.711218       -0.003766           0.468889            0.104731     155905.0       256.0        10.0       0.005779         0.753059            6.0       seed7        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16   123    0.005746        1.091569       0.709011    0.012611 -0.132852            0.490278       0.463333              0.455833                    0.474444        0.067440         0.001911           0.007359         0.214722 584585.766824        532931.666667       1.096924              True        0.005264            0.008105           0.690610       -0.017010           0.511111            0.048674     155905.0       256.0        10.0       0.005576         0.657379            6.0     seed123        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16  2026    0.005812        1.104146       0.717180    0.008985 -0.148827            0.495000       0.486667              0.455833                    0.474444        0.125582         0.002608           0.007359         0.204444 590335.220283        532931.666667       1.107713              True        0.005264            0.008105           0.695789        0.024992           0.514444            0.089370     155905.0       256.0        10.0       0.005672         0.566638            5.9    seed2026        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure ITransformerLike     huber seasonal_diff16 window_standard       16    -1    0.005480        1.040963       0.676141    0.040235 -0.054972            0.497222       0.472222              0.455833                    0.474444        0.041470         0.001499           0.007359         0.317222 556268.355075        532931.666667       1.043789             False        0.005264            0.008105                NaN             NaN                NaN                 NaN          NaN         NaN         NaN            NaN              NaN            NaN ensemble(4)         True                0.021755                   0.487500            0.018480          -0.015278
t6_signal_boost coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16 window_standard       16    42    0.005484        1.041846       0.676714    0.017420 -0.063558            0.512778       0.490000              0.455833                    0.474444        0.069055         0.001934           0.007359         0.301389 555901.241661        532931.666667       1.043100              True        0.005264            0.008105           0.686632       -0.054532           0.441111            0.054177     162049.0       256.0        10.0       0.005584         0.216615            5.9      seed42        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16 window_standard       16     7    0.005508        1.046415       0.679682   -0.033789 -0.070607            0.498889       0.476667              0.455833                    0.474444        0.046636         0.001589           0.007359         0.339444 558808.581989        532931.666667       1.048556             False        0.005264            0.008105           0.674552       -0.039578           0.484444            0.033431     162049.0       256.0        10.0       0.005485         0.183023            6.0       seed7        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16 window_standard       16   123    0.005482        1.041302       0.676361    0.010230 -0.058172            0.504722       0.513333              0.455833                    0.474444        0.062316         0.001837           0.007359         0.322500 555337.616875        532931.666667       1.042043              True        0.005264            0.008105           0.680660       -0.039168           0.477778            0.044704     162049.0       256.0        10.0       0.005501         0.192402            6.0     seed123        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16 window_standard       16  2026    0.005512        1.046999       0.680061    0.002911 -0.066595            0.501111       0.496667              0.455833                    0.474444        0.060684         0.001813           0.007359         0.321111 558547.593671        532931.666667       1.048066              True        0.005264            0.008105           0.673751        0.045810           0.513333            0.046362     162049.0       256.0        10.0       0.005425         0.307180            5.8    seed2026        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure     PatchTSTLike     huber seasonal_diff16 window_standard       16    -1    0.005356        1.017439       0.660861    0.000861 -0.030064            0.517222       0.502222              0.455833                    0.474444        0.025645         0.001179           0.007359         0.472500 542305.961588        532931.666667       1.017590             False        0.005264            0.008105                NaN             NaN                NaN                 NaN          NaN         NaN         NaN            NaN              NaN            NaN ensemble(4)         True               -0.000807                   0.494167            0.001668           0.008056
t6_signal_boost coin_multitimeframe_structure           Linear     huber seasonal_diff16 window_standard       16    42    0.005693        1.081499       0.702470    0.001821 -0.112599            0.503333       0.497778              0.455833                    0.474444        0.106158         0.002398           0.007359         0.243333 577921.173888        532931.666667       1.084419              True        0.005264            0.008105           0.688418       -0.013384           0.475556            0.074534      98497.0       256.0        10.0       0.005658         0.215451            3.0      seed42        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure           Linear     huber seasonal_diff16 window_standard       16     7    0.005658        1.074770       0.698100    0.040548 -0.095953            0.524722       0.537778              0.455833                    0.474444        0.122635         0.002577           0.007359         0.234444 574780.350657        532931.666667       1.078525              True        0.005264            0.008105           0.698442       -0.003280           0.497778            0.087025      98497.0       256.0        10.0       0.005718         0.254980            3.0       seed7        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure           Linear     huber seasonal_diff16 window_standard       16   123    0.006381        1.212211       0.787372    0.039352 -0.321076            0.514722       0.518889              0.455833                    0.474444        0.359840         0.004415           0.007359         0.138611 649638.744398        532931.666667       1.218991              True        0.005264            0.008105           0.772919        0.015355           0.501111            0.270488      98497.0       256.0        10.0       0.006392         0.280624            3.0     seed123        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure           Linear     huber seasonal_diff16 window_standard       16  2026    0.005688        1.080445       0.701786    0.030359 -0.107076            0.513333       0.501111              0.455833                    0.474444        0.109869         0.002439           0.007359         0.232778 577818.619607        532931.666667       1.084226              True        0.005264            0.008105           0.700020       -0.009262           0.511111            0.082631      98497.0       256.0        10.0       0.005721         0.226739            3.0    seed2026        False                     NaN                        NaN                 NaN                NaN
t6_signal_boost coin_multitimeframe_structure           Linear     huber seasonal_diff16 window_standard       16    -1    0.005509        1.046516       0.679747    0.043758 -0.060190            0.513611       0.510000              0.455833                    0.474444        0.076662         0.002038           0.007359         0.298611 558909.325760        532931.666667       1.048745              True        0.005264            0.008105                NaN             NaN                NaN                 NaN          NaN         NaN         NaN            NaN              NaN            NaN ensemble(4)         True                0.028020                   0.513889            0.015737          -0.003889
```

## seed ensemble 이득 (음수면 강화 실패)

```
           model  single_mean_trend_corr  trend_corr  ensemble_corr_gain  ensemble_lda_gain
ITransformerLike                  0.0218      0.0402              0.0185            -0.0153
    PatchTSTLike                 -0.0008      0.0009              0.0017             0.0081
          Linear                  0.0280      0.0438              0.0157            -0.0039
```
