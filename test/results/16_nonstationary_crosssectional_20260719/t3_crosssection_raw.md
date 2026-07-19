# 16번 raw 결과 — t3_crosssection

- 실행 시각: 2026-07-19 16:52:19 (서버, 헤드리스 .py)
- 드라이버: `test/models/16_nonstationary_crosssectional_test.py` --suite t3_crosssection
- 케이스 수: 30 (실패 0)

## 실행 인자

```json
{
  "suite": "t3_crosssection",
  "db": null,
  "profile": "school_4090_15gb",
  "device": null,
  "num_workers": 0,
  "models": "ITransformerLike",
  "normalizations": "window_standard,revin,dishts_lite,none",
  "losses": "huber,pinball,student_t,tail_weighted",
  "seeds": "42",
  "normalization": "revin",
  "loss": "huber",
  "preprocessing": "seasonal_diff16",
  "horizon": 16,
  "n_tickers": 30,
  "seq_len": 64,
  "hidden": 96,
  "epochs": 10,
  "patience": 4,
  "min_delta": 1e-05,
  "batch_size": 256,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "gradient_policy": "clip1",
  "lr": 0.001,
  "weight_decay": 0.0001,
  "max_rows": null,
  "max_windows": 12000,
  "stride": 3,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
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
  "memory_available_gb": 24.329208374023438,
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
  "rows": 104736,
  "start": "2023-07-20 22:30:00",
  "end": "2026-07-18 22:30:00",
  "close_mean": 2050.64294702872,
  "close_std": 1307.504749143277,
  "close_min": 564.6,
  "close_max": 4973.0,
  "return_mean": 3.538751310972873e-06,
  "return_std": 0.004150658643807229,
  "return_skew": -0.3736980514939095,
  "return_kurtosis": 158.51104664816927,
  "missing_cells": 0
}
```

## 이 suite가 보는 것 — KRW 전 종목 pooled 학습

- **고정**: 우승 정규화·손실, h=16.
- **변화**: 학습 축 {single(BTC), pooled_ci(전종목 채널독립), pooled_cd}.
- **질문**: 전 종목을 함께 학습하면 신호가 종목 가로질러 일반화되는가(BTC 우연 탈피)?
- **읽는 법**: pooled 모델의 종목별 trend_corr가 고루 양수면 구조적 신호.

## 지표 사전

- `trend_corr`: 예측 vs 실제 h-step 수익률 Pearson 상관. 0=무상관. 금융에선 0.1도 유의미.
- `variance_ratio`: 예측분산/실제분산. 1이 이상적(실제만큼 출렁임), ≪0.1 평탄화, ≫20 폭주. '진폭' 지표.
- `large_move_da`: |실제| 상위 25%(큰 변동) 구간 방향 정확도. 0.5=동전던지기. 큰 변동 포착 직접 지표.
- `tail_precision`: 큰 변동을 '예측이 큰 변동이라 본 것' 중 실제 큰 변동 비율(정밀도).
- `tail_recall`: 실제 큰 변동 중 예측이 잡아낸 비율(재현율). 작은 폭만 예측하면 여기서 낮게 나옴.
- `tail_f1`: tail 정밀도·재현율의 조화평균. 큰 변동 포착의 종합 지표(정확도 대신 이걸 본다).
- `mase_momentum`: 예측 오차/'직전 h봉 추세 지속' 오차. <1이면 추세지속 기준선보다 우수.
- `copy_risk_krw`: KRW MAE/persistence MAE. <1이면 직전값 복사보다 우수.
- `cross_sectional_ic`: 같은 시점 종목 간 예측-실제 순위상관(Spearman) 평균. 상대강도 신호 지표.
- `val_overfit_gap`: best val_loss 대비 마지막 val_loss 상승분. 과적합 진단(양수 크면 과적합).
- `healthy_variance`: variance_ratio가 0.05~20이면 True(평탄화/폭주 아님).

## 전체 leaderboard

```
          suite    ticker            model  loss normalization   preprocessing  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr  direction_accuracy  large_move_da  variance_ratio  pred_return_std  actual_return_std  copy_risk_krw  healthy_variance  tail_precision  tail_recall  tail_f1  val_overfit_gap  horizon   rows
t3_crosssection   KRW-XRP ITransformerLike huber         revin seasonal_diff16    42    0.016291        2.160738       1.440705   -0.004994            0.562222       0.577778        0.577580         0.007924           0.010427       2.129999              True        0.255556     0.255556 0.255556         0.000000       16 104736
t3_crosssection   KRW-BTC ITransformerLike huber         revin seasonal_diff16    42    0.010116        1.753920       1.146690    0.027251            0.519444       0.555556        0.497812         0.005624           0.007971       1.748358              True        0.208889     0.208889 0.208889         0.000000       16 104768
t3_crosssection  KRW-DOGE ITransformerLike huber         revin seasonal_diff16    42    0.012547        1.395802       0.869044   -0.028531            0.438333       0.444934        0.511520         0.008614           0.012044       1.401117              True        0.233333     0.231278 0.232301         0.001320       16 104573
t3_crosssection   KRW-ETH ITransformerLike huber         revin seasonal_diff16    42    0.013857        1.786381       1.139203   -0.006276            0.482778       0.448889        1.235273         0.012289           0.011057       1.807517              True        0.255556     0.255556 0.255556         0.005581       16 104739
t3_crosssection   KRW-SOL ITransformerLike huber         revin seasonal_diff16    42    0.010970        1.195250       0.794131    0.009936            0.527778       0.520000        0.233268         0.006025           0.012475       1.197985              True        0.244444     0.244444 0.244444         0.003249       16 104759
t3_crosssection  KRW-USDT ITransformerLike huber         revin seasonal_diff16    42    0.005834        4.055157       2.644412    0.082208            0.603889       0.543237        3.232615         0.003616           0.002011       4.039902              True        0.237778     0.237251 0.237514         0.001143       16  73550
t3_crosssection  KRW-SHIB ITransformerLike huber         revin seasonal_diff16    42    0.012515        1.564262       1.031367   -0.016432            0.568333       0.548889        0.553580         0.007953           0.010689       1.547672              True        0.251111     0.251111 0.251111         0.003428       16 103386
t3_crosssection   KRW-SEI ITransformerLike huber         revin seasonal_diff16    42    0.014558        1.098205       0.754014    0.057296            0.520000       0.528889        0.234573         0.008397           0.017338       1.091622              True        0.240000     0.240000 0.240000         0.020213       16 100136
t3_crosssection   KRW-XLM ITransformerLike huber         revin seasonal_diff16    42    0.018293        1.076151       0.741392    0.007639            0.590556       0.475556        0.026335         0.004244           0.026152       1.064046             False        0.226667     0.226667 0.226667         0.001559       16 103263
t3_crosssection   KRW-SUI ITransformerLike huber         revin seasonal_diff16    42    0.017595        1.640310       1.054872   -0.033781            0.516667       0.533333        0.769591         0.012755           0.014540       1.610738              True        0.262222     0.262222 0.262222         0.003861       16 104657
t3_crosssection   KRW-ETC ITransformerLike huber         revin seasonal_diff16    42    0.011894        1.433359       0.963590   -0.036852            0.453333       0.454545        0.435174         0.007278           0.011032       1.435993              True        0.228889     0.228381 0.228635         0.000000       16 104258
t3_crosssection   KRW-ADA ITransformerLike huber         revin seasonal_diff16    42    0.017199        1.518081       1.023962   -0.000798            0.408333       0.451111        0.307610         0.008907           0.016059       1.553216              True        0.246667     0.246667 0.246667         0.001032       16 104261
t3_crosssection   KRW-STX ITransformerLike huber         revin seasonal_diff16    42    0.014779        1.216123       0.800267   -0.024261            0.590556       0.573333        0.166775         0.006490           0.015891       1.199698              True        0.271111     0.271111 0.271111         0.000000       16 102434
t3_crosssection  KRW-ONDO ITransformerLike huber         revin seasonal_diff16    42    0.018906        1.257656       0.838311   -0.039824            0.491667       0.475556        0.357843         0.012416           0.020756       1.249508              True        0.275556     0.275556 0.275556         0.004422       16  72892
t3_crosssection  KRW-HBAR ITransformerLike huber         revin seasonal_diff16    42    0.012495        1.206337       0.761009   -0.023098            0.483333       0.530973        0.214080         0.007109           0.015366       1.198140              True        0.242222     0.241150 0.241685         0.002951       16 101103
t3_crosssection   KRW-ENS ITransformerLike huber         revin seasonal_diff16    42    0.014636        1.268353       0.824934    0.041358            0.538889       0.535556        0.317172         0.008983           0.015950       1.253482              True        0.264444     0.264444 0.264444         0.002232       16  69718
t3_crosssection   KRW-GAS ITransformerLike huber         revin seasonal_diff16    42    0.014395        1.381147       0.917557   -0.016550            0.528333       0.531111        0.470200         0.010157           0.014813       1.377619              True        0.235556     0.235556 0.235556         0.003943       16 101411
t3_crosssection  KRW-NEAR ITransformerLike huber         revin seasonal_diff16    42    0.021056        1.175653       0.801307   -0.066397            0.513333       0.482222        0.116599         0.008478           0.024828       1.160865              True        0.264444     0.264444 0.264444         0.003090       16 103274
t3_crosssection   KRW-BCH ITransformerLike huber         revin seasonal_diff16    42    0.013565        1.161237       0.763570   -0.022647            0.521111       0.522222        0.187495         0.007163           0.016542       1.153548              True        0.264444     0.264444 0.264444         0.000000       16 104181
t3_crosssection  KRW-SAND ITransformerLike huber         revin seasonal_diff16    42    0.014657        1.250147       0.814865    0.077043            0.569444       0.575556        0.279631         0.008445           0.015970       1.241800              True        0.277778     0.277778 0.277778         0.003612       16 100511
t3_crosssection  KRW-MOVE ITransformerLike huber         revin seasonal_diff16    42    0.019899        1.296595       0.853188    0.055545            0.486667       0.491111        0.336111         0.013924           0.024018       1.294012              True        0.213333     0.213333 0.213333         0.003897       16  52498
t3_crosssection   KRW-ARK ITransformerLike huber         revin seasonal_diff16    42    0.015566        1.070693       0.720028    0.001302            0.509444       0.503311        0.088806         0.006283           0.021083       1.076644              True        0.295556     0.293598 0.294574         0.005236       16  92957
t3_crosssection KRW-POLYX ITransformerLike huber         revin seasonal_diff16    42    0.017644        1.188692       0.801941   -0.050011            0.505556       0.506667        0.246487         0.010202           0.020549       1.182343              True        0.242222     0.242222 0.242222         0.006286       16  93425
t3_crosssection  KRW-LINK ITransformerLike huber         revin seasonal_diff16    42    0.012325        1.389685       0.897615    0.068950            0.525000       0.553333        0.349535         0.006990           0.011823       1.379268              True        0.280000     0.280000 0.280000         0.000000       16 104445
t3_crosssection  KRW-BLUR ITransformerLike huber         revin seasonal_diff16    42    0.018623        1.046354       0.706751    0.014849            0.569444       0.545455        0.082457         0.008428           0.029350       1.048224              True        0.255556     0.254989 0.255272         0.002134       16  94599
t3_crosssection   KRW-CTC ITransformerLike huber         revin seasonal_diff16    42    0.016018        1.210477       0.772659   -0.001030            0.500000       0.502222        0.305610         0.010149           0.018358       1.228521              True        0.248889     0.248889 0.248889         0.007838       16  83308
t3_crosssection KRW-WAVES ITransformerLike huber         revin seasonal_diff16    42    0.015853        1.260458       0.806333    0.026095            0.596111       0.623060        0.083680         0.006246           0.021590       1.257441              True        0.233333     0.232816 0.233074         0.000000       16  99639
t3_crosssection KRW-AERGO ITransformerLike huber         revin seasonal_diff16    42    0.022447        1.199437       0.745817    0.000615            0.492778       0.535556        0.090416         0.011273           0.037491       1.207916              True        0.237778     0.237778 0.237778         0.009278       16  91379
t3_crosssection  KRW-AVAX ITransformerLike huber         revin seasonal_diff16    42    0.013057        1.269757       0.822378   -0.022343            0.522778       0.511111        0.280430         0.007512           0.014186       1.267905              True        0.271111     0.271111 0.271111         0.004455       16 103802
t3_crosssection KRW-STRAX ITransformerLike huber         revin seasonal_diff16    42    0.023099        1.111112       0.741439    0.005542            0.460556       0.483370        0.040633         0.008553           0.042430       1.110307             False        0.282222     0.281596 0.281909         0.000000       16  94450
```

## cross-sectional 일반화 요약

30종목 중 trend_corr>0: 14 (47%). **cross_sectional_ic = -0.0139** (같은 시점 종목 간 예측-실제 Spearman 평균; 양수면 상대강도 신호 존재). tail_f1 평균 0.251, variance_ratio 평균 0.421.
