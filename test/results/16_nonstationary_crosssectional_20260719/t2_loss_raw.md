# 16번 raw 결과 — t2_loss

- 실행 시각: 2026-07-19 16:51:10 (서버, 헤드리스 .py)
- 드라이버: `test/models/16_nonstationary_crosssectional_test.py` --suite t2_loss
- 케이스 수: 4 (실패 0)

## 실행 인자

```json
{
  "suite": "t2_loss",
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
  "max_windows": 16000,
  "stride": 2,
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
  "memory_available_gb": 24.34961700439453,
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

## 이 suite가 보는 것 — 큰 변동 손실함수 비교

- **고정**: 우승 정규화, 모델, BTC 단일축, h=16.
- **변화**: 손실 {huber, pinball(다중분위), student_t(분포), tail_weighted}.
- **질문**: 평균형 huber를 분위·분포·tail 손실로 바꾸면 큰 변동(tail)을 더 잘 잡는가?
- **읽는 법**: tail_f1·trend_corr가 오르고 variance_ratio가 1에 가까워지면 큰 변동 포착 개선.

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
  suite  ticker            model          loss normalization   preprocessing  seed  mae_return  mae_zero_ratio  mase_momentum  trend_corr  direction_accuracy  large_move_da  variance_ratio  pred_return_std  actual_return_std  copy_risk_krw  healthy_variance  tail_precision  tail_recall  tail_f1  val_overfit_gap  horizon  champion_score
t2_loss KRW-BTC ITransformerLike         huber         revin seasonal_diff16    42    0.010291        1.720449       1.124257    0.025736            0.515000       0.526667        1.550159         0.010234            0.00822       1.722440              True        0.205000     0.205000 0.205000         0.001102       16          0.1659
t2_loss KRW-BTC ITransformerLike       pinball         revin seasonal_diff16    42    0.008497        1.420629       0.928334   -0.035920            0.487083       0.456667        0.667633         0.006716            0.00822       1.426179              True        0.243333     0.243333 0.243333         0.001880       16          0.1394
t2_loss KRW-BTC ITransformerLike     student_t         revin seasonal_diff16    42    0.017680        2.955781       1.931505   -0.036591            0.472500       0.445000        2.234798         0.012288            0.00822       2.998338              True        0.240000     0.240000 0.240000         0.073880       16          0.0644
t2_loss KRW-BTC ITransformerLike tail_weighted         revin seasonal_diff16    42    0.007514        1.256154       0.820855    0.069005            0.529167       0.550000        0.154351         0.003229            0.00822       1.250982              True        0.231667     0.231667 0.231667         0.000000       16          0.0014
```

## 실행 중 이슈·판단 로그 (raw, 취사선택은 사용자 몫)

- t2_loss 우승 선정: huber (tail_f1=0.205, large_move_da=0.527, variance_ratio=1.550, score=0.166)
