# 13번 feature-algorithm-resource 실험 — 캡처 전량 통합 결과 (full captured dump)

> 작성일: 2026-06-28
> 대상 노트북: `test/models/13_feature_algorithm_resource_test.ipynb`
> 본 문서는 **저장에 실패한 13번 노트북**에서 복원 가능한 모든 출력을, 화면 캡처
> 4장(`test/images/13_output_recovery_20260627/capture_001~004.png`)을 직접 판독해
> 한 파일에 빠짐없이 옮긴 통합본이다.

---

## 0. 이 문서의 성격과 한계 (먼저 읽을 것)

### 0.1 왜 이 문서가 "원본 대체물"이 되는가

13번 노트북은 실행 중 두 장애가 겹쳐 결과가 디스크에 저장되지 못했다.

1. **실행 장애**: PyTorch `DataLoader`가 worker process로 batch를 넘기며 공유 메모리
   (`/dev/shm`)를 쓰는데 서버 한도에 걸려
   `RuntimeError: unable to allocate shared memory(shm) ... Resource temporarily unavailable (11)`
   로 중단됐다.
2. **저장 장애**: 노트북 output 누적량이 너무 커서 VSCode가
   `Error: Notebook too large to backup`을 반복하며 snapshot/backup 생성에 실패했다.
   그 결과 일반 저장, 다른 이름 저장, hot-exit backup이 모두 불안정해졌다.

현재 디스크의 `.ipynb`는 약 `40,881 bytes`의 실행 엔트리포인트 상태이며,
저장된 `image/png` 0개, `stream` 0개, `error` 0개다. 즉 결과 원본 역할을 못 한다.

따라서 **화면에 떠 있던 output을 캡처한 PNG 4장이 유일한 source of truth**이며,
이 문서는 그 4장을 글자 단위로 옮긴 복원본이다.

### 0.2 무엇이 복원되고 무엇이 영구 소실인가

| 구분 | 항목 | 상태 |
|---|---|---|
| 복원됨 | fusion 전략 표 (5행 중 3행 가시) | 캡처 1 |
| 복원됨 | case 28/1440 전체 epoch 로그 (1~14, early-stop) | 캡처 1 |
| 복원됨 | MambaLike__seasonal_diff16__balanced_composite__seed42 진단 그래프 6종 | 캡처 1·3 |
| 복원됨 | risk branch 그래프 3종 + event preview 10행 | 캡처 2 |
| 복원됨 | point prediction preview 10행 (actual OHLC) | 캡처 3 |
| 복원됨 | 고스케일 학습 로그 (loss ≈0.9, epoch 1~7) | 캡처 3 |
| 복원됨 | 저스케일 학습 로그 (epoch 8~24) | 캡처 4 |
| 복원됨 | PatchTSTLike__winsor_025__balanced_composite__seed2026 진단 그래프 6종 | 캡처 4 |
| **소실** | 전체 1440 케이스 leaderboard | 캡처에 없음 |
| **소실** | feature group 평균 요약 | 캡처에 없음 |
| **소실** | model family 평균 요약 | 캡처에 없음 |
| **소실** | feature × model heatmap | 캡처에 없음 |
| **소실** | final decision / top candidates 요약 | 캡처에 없음 |
| **소실** | 마지막 full traceback 전체 | 일부만 가시(캡처 2) |
| **소실** | 마지막까지 성공한 case 범위 | 캡처에 없음 |

소실 항목은 디스크·VSCode local history·캡처 어디에도 없으므로 **재실행 외에는 복원 불가**다.
이 문서는 그 사실을 숨기지 않고, 없는 표를 추정으로 채우지 않는다.

---

## 1. 실행 및 분석 환경

- 커널: `Python 3.12 (Quant Stat)`
  - 가상환경: `quant_uv_py312_20260614_045930`
- 실행 서버 경로(traceback 기준):
  `/home/std_jun99120/personal_ai_project/quantitative_trading/`
- 노트북 상단 마커: `# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]`
- 데이터: KRW-BTC 캔들 (preview의 close 수준 ≈ `1.14e8` KRW, 2026-05-20 구간)
- 캡처 로그 경로:
  - Jupyter: `%APPDATA%\Code\logs\20260624T203952\window1\exthost\output_logging_20260624T203954\7-Jupyter.log`
  - VSCode renderer: `%APPDATA%\Code\logs\20260624T203952\window1\renderer.log`

---

## 2. 확인된 실험 축 (캡처에서 실제 관찰된 조합)

캡처에 등장한 케이스 라벨에서 다음 축이 실제로 실행되었음이 확정된다.

- **총 케이스 수**: `1440` (case 28/1440 로그 헤더 기준)
- **feature set**: `mtf_returns_only`
- **preprocessing**: `seasonal_diff16`, `winsor_025`
- **point models**: `MambaLike`, `PatchTSTLike`, `TCN`
- **fusion(point+risk)**: `MambaLike + PatchTSTLike`
- **risk target**: `absolute_move`
- **horizon**: `horizon16`
- **objective**: `balanced_composite`
- **seeds**: `42`, `777`, `2026`
- **seq length**: `seq64`

캡처에서 직접 읽힌 케이스 라벨 원문:

1. `[case 28/1440] mtf_returns_only / seasonal_diff16 / MambaLike + PatchTSTLike / seed42 / seq64 / ab...` (이하 잘림)
2. `MambaLike__seasonal_diff16__balanced_composite__seed42` (진단 그래프 + prediction preview)
3. `PatchTSTLike__mtf_returns_only__winsor_025__absolute_move__horizon16__seed777` (risk event preview)
4. `mtf_returns_only / winsor_025 / TCN + PatchTSTLike / seed777` (캡처 2 하단, 잘림)
5. `PatchTSTLike__winsor_025__balanced_composite__seed2026` (진단 그래프)

---

## 3. Fusion 전략 표 (캡처 1 상단) — 원문 그대로

표 메타: `5 rows x 8 cols`, `Page 1 of 1`, `10 per page`
화면에 가시였던 행은 2~4번(컬럼 1~4까지). 1번 행과 5번째 행, 그리고 5~8번 컬럼은 화면 밖이라 미가시.

| # | 전략 | col1 | col2 | col3 | col4 |
|---|---|---|---|---|---|
| 2 | `risk_gate_only` | -0.9728078171607566 | -0.9729251501572636 | -0.14879707862244232 | 0.5721652766282549 |
| 3 | `point_plus_risk_gate` | -0.9880838073810395 | -0.9880795475592111 | -0.21292424496937234 | 0.25600050932705165 |
| 4 | `no_trade` | 0.0 | 0.0 | 0.0 | 0.0 |

> 컬럼 헤더는 화면에 가려져 라벨을 읽지 못했다. 값의 부호/규모로 보아 col1·col2는
> 손실 또는 음의 수익 계열 지표(클수록 나쁨), col4는 양의 보조 지표로 추정되나
> **헤더 미확인이므로 지표명은 확정하지 않는다.** `no_trade`가 전부 0.0인 것은
> "거래 안 함 = 기준선 0" 정의와 일치한다.

---

## 4. case 28/1440 전체 학습 로그 (캡처 1) — 원문 그대로

케이스: `mtf_returns_only / seasonal_diff16 / MambaLike + PatchTSTLike / seed42 / seq64`

```text
epoch=001 train=0.027779 val=0.007803 grad=0.7681 lr=9.96e-04
epoch=002 train=0.006861 val=0.004385 grad=1.4563 lr=9.83e-04
epoch=003 train=0.005827 val=0.004261 grad=2.2450 lr=9.62e-04
epoch=004 train=0.005335 val=0.003670 grad=2.4762 lr=9.33e-04
epoch=005 train=0.004233 val=0.006847 grad=2.0772 lr=8.97e-04
epoch=006 train=0.005724 val=0.006942 grad=2.1944 lr=8.54e-04
epoch=007 train=0.005047 val=0.002502 grad=2.0370 lr=8.04e-04
epoch=008 train=0.004877 val=0.006243 grad=2.9462 lr=7.50e-04
epoch=009 train=0.004798 val=0.006004 grad=5.2536 lr=6.91e-04
epoch=010 train=0.007392 val=0.017507 grad=6.0345 lr=6.29e-04
epoch=011 train=0.006018 val=0.005645 grad=4.7306 lr=5.65e-04
epoch=012 train=0.005519 val=0.003893 grad=2.7140 lr=5.00e-04
epoch=013 train=0.003264 val=0.003868 grad=2.4865 lr=4.35e-04
epoch=014 train=0.003962 val=0.004930 grad=2.8744 lr=3.71e-04
[early-stop] epoch=14 best_val=0.002502
```

관찰:
- best validation은 **epoch 7의 val=0.002502**에서 나왔고, 이후 7 epoch 동안 갱신이 없어
  early-stop이 epoch 14에서 발동했다(patience≈7로 추정).
- epoch 10에서 val이 `0.017507`로 튀고 동시에 grad가 `6.0345`로 커졌다 →
  국소적 불안정 구간이 있었으나 회복했다.
- lr은 cosine 계열로 9.96e-04 → 3.71e-04까지 단조 감소.

---

## 5. 추가 학습 로그 2건 (캡처 3·4)

### 5.1 고스케일 로그 (loss ≈ 0.9 단위, 캡처 3 하단)

이 로그는 §4·§5.2와 손실 스케일이 완전히 다르다(0.9 vs 0.003). 서로 다른 objective/대상으로 보인다.

```text
epoch=001 train=0.945247 val=0.882157 grad=0.2791
epoch=002 train=0.926420 val=0.861249 grad=0.2542
epoch=003 train=0.912067 val=0.904456 grad=0.3113
epoch=004 train=0.898672 val=0.939426 grad=0.3718
epoch=005 train=0.881307 val=0.883570 grad=0.4036
epoch=006 train=0.866108 val=0.927276 grad=0.4509
epoch=007 train=0.848609 val=0.933704 grad=0.5264
```
> epoch 7까지만 화면에 가시. train은 단조 감소하나 val은 0.86~0.94 사이에서
> 진동하며 거의 줄지 않는다 → 이 분기는 학습이 일반화로 이어지지 않는 신호.

### 5.2 저스케일 로그 epoch 8~24 (캡처 4 상단)

§4와 동일 스케일(0.001~0.004)이나 epoch 24까지 이어지는 별도/연장 케이스로 보인다.

```text
epoch=008 train=0.004546 val=0.002000 grad=1.1591 lr=7.50e-04
epoch=009 train=0.003754 val=0.002662 grad=0.8885 lr=6.91e-04
epoch=010 train=0.003674 val=0.003145 grad=1.3215 lr=6.29e-04
epoch=011 train=0.004252 val=0.003131 grad=2.5832 lr=5.65e-04
epoch=012 train=0.002965 val=0.001711 grad=1.1004 lr=5.00e-04
epoch=013 train=0.002959 val=0.001571 grad=1.0143 lr=4.35e-04
epoch=014 train=0.003349 val=0.003557 grad=0.8915 lr=3.71e-04
epoch=015 train=0.003022 val=0.001793 grad=1.6795 lr=3.09e-04
epoch=016 train=0.002698 val=0.001756 grad=1.4944 lr=2.50e-04
epoch=017 train=0.002561 val=0.001752 grad=2.5151 lr=1.96e-04
epoch=018 train=0.002096 val=0.001385 grad=0.5804 lr=1.46e-04
epoch=019 train=0.002003 val=0.001245 grad=2.8150 lr=1.03e-04
epoch=020 train=0.001939 val=0.001136 grad=1.1100 lr=6.70e-05
epoch=021 train=0.001954 val=0.001530 grad=0.5652 lr=3.81e-05
epoch=022 train=0.001900 val=0.001245 grad=10.5036 lr=1.70e-05
epoch=023 train=0.001866 val=0.001118 grad=2.3547 lr=4.28e-06
epoch=024 train=0.001869 val=0.001120 grad=3.8322 lr=0.00e+00
```
관찰:
- best val은 **epoch 23의 val=0.001118** 부근. epoch 8~24에서 안정적으로 수렴.
- **epoch 22에서 grad=10.5036 단일 스파이크** 발생(lr이 1.70e-05로 거의 0인데도 발생) →
  optimization stability 점검 대상. lr이 작아 손실에는 큰 영향 없었음.

---

## 6. risk branch — event preview 표 (캡처 2)

케이스: `[PatchTSTLike__mtf_returns_only__winsor_025__absolute_move__horizon16__seed777 event preview]`
표 메타: `12 rows x 8 cols`, `Page 1 of 2` (즉 전체 12행 중 10행 가시, 11~12행은 2페이지)

| # | timestamp | prev_close | future_end_close | future_return | event_score |
|---|---|---|---|---|---|
| 0 | 2026-05-20 00:15:00 | 1.14554e+08 | 1.14106e+08 | -0.003918 | 0.006165 |
| 1 | 2026-05-20 00:30:00 | 1.14251e+08 | 1.14061e+08 | -0.001664 | 0.003516 |
| 2 | 2026-05-20 00:45:00 | 1.14307e+08 | 1.14093e+08 | -0.001874 | 0.004006 |
| 3 | 2026-05-20 01:00:00 | 1.14277e+08 | 1.13968e+08 | -0.002708 | 0.003744 |
| 4 | 2026-05-20 01:15:00 | 1.144e+08   | 1.14162e+08 | -0.002083 | 0.004819 |
| 5 | 2026-05-20 01:30:00 | 1.14515e+08 | 1.14342e+08 | -0.001512 | 0.005824 |
| 6 | 2026-05-20 01:45:00 | 1.14614e+08 | 1.14541e+08 | -0.000637 | 0.006688 |
| 7 | 2026-05-20 02:00:00 | 1.1442e+08  | 1.14619e+08 |  0.001738 | 0.004994 |
| 8 | 2026-05-20 02:15:00 | 1.14136e+08 | 1.14536e+08 |  0.003498 | 0.004223 |
| 9 | 2026-05-20 02:30:00 | 1.14402e+08 | 1.14647e+08 |  0.002139 | 0.004837 |

> `event_label` 컬럼은 헤더만 가시하고 값은 화면 밖. future_return은 ±0.4% 내 소폭,
> event_score는 0.0035~0.0067 범위로 비교적 평탄하다(이 구간엔 강한 이벤트가 없음).

### 6.1 risk branch 진단 그래프 3종 (캡처 2 하단)

1. **Precision-recall curve** — x: recall(0~1), y: precision(0~0.6).
   - 관찰: recall 0 근처에서 precision≈0.6로 시작해 빠르게 0.3 기준선(점선)으로 수렴.
   - 해석: 양성(이벤트) 분류기의 판별력이 약하다. 기준선 0.3은 base rate로 추정.
   - 좋음/나쁨: **나쁨 쪽**. recall을 조금만 올려도 precision이 base rate로 붕괴.
2. **Selective risk-coverage** — x: fraction of samples accepted(0~1), y: classification error(0.30~0.42).
   - 관찰: 수용 비율이 늘수록 error가 0.30→0.42로 단조 증가.
   - 해석: 가장 확신하는 샘플만 받으면 error가 낮다(선택적 예측이 작동).
   - 좋음/나쁨: **부분적으로 좋음**. selective prediction의 단조성은 정상이나 절대 error가 높다.
3. **Probability versus realized risk** — x: realized future risk score, y: predicted event probability.
   - 관찰: 산점이 넓게 퍼져 양의 상관이 약하게만 보임.
   - 해석: 예측 확률과 실제 위험의 정렬(calibration)이 약하다.

### 6.2 캡처 2 상단 부분 그래프 + traceback

- 상단에 weight/epoch 감소 곡선, test time index에 따른 신호, mean predicted probability vs obs 산점이 부분 가시(라벨 잘림).
- 표 하단 traceback(부분):
  ```text
  /home/std_jun99120/personal_ai_project/quantitative_trading/test/models/9_preprocessing_uncertaint...
      fig.tight_layout()
  /home/std_jun99120/personal_ai_project/quantitative_trading/.venvs/quant_uv_py312_20260614_045930/...
      fig.canvas.print_figure(bytes_io, **kw)
  ```
  → matplotlib figure를 PNG로 렌더(`print_figure`)하는 단계에서 발생한 흐름. 9번 전처리
  노트북에서 가져온 진단 플롯 유틸 경로가 보인다. **전체 traceback 마지막 줄은 미가시.**

---

## 7. point branch — prediction preview 표 (캡처 3)

케이스: `[MambaLike__seasonal_diff16__balanced_composite__seed42 prediction preview]`
표 메타: `12 rows x 10 cols`, `Page 1 of 2` (10행 가시). 예측 OHLC 컬럼(`pred_open_...` 이후)은 화면 밖.

| # | timestamp | actual_open | actual_high | actual_low | actual_close |
|---|---|---|---|---|---|
| 0 | 2026-05-20 00:30:00 | 1.14109e+08 | 1.14251e+08 | 1.14017e+08 | 1.14061e+08 |
| 1 | 2026-05-20 00:45:00 | 1.14061e+08 | 1.14193e+08 | 1.13987e+08 | 1.14093e+08 |
| 2 | 2026-05-20 01:00:00 | 1.14116e+08 | 1.14144e+08 | 1.13933e+08 | 1.13968e+08 |
| 3 | 2026-05-20 01:15:00 | 1.13935e+08 | 1.14162e+08 | 1.1393e+08  | 1.14162e+08 |
| 4 | 2026-05-20 01:30:00 | 1.14161e+08 | 1.14342e+08 | 1.14148e+08 | 1.14342e+08 |
| 5 | 2026-05-20 01:45:00 | 1.14343e+08 | 1.14543e+08 | 1.14343e+08 | 1.14541e+08 |
| 6 | 2026-05-20 02:00:00 | 1.14542e+08 | 1.14647e+08 | 1.14491e+08 | 1.14619e+08 |
| 7 | 2026-05-20 02:15:00 | 1.1462e+08  | 1.14762e+08 | 1.14536e+08 | 1.14536e+08 |
| 8 | 2026-05-20 02:30:00 | 1.14525e+08 | 1.1472e+08  | 1.14406e+08 | 1.14647e+08 |
| 9 | 2026-05-20 02:45:00 | 1.14573e+08 | 1.14624e+08 | 1.14469e+08 | 1.1455e+08  |

> 예측 OHLC(`pred_open_proxy` 등)는 2페이지/화면 밖이라 미확보. actual OHLC는
> KRW-BTC 1분/15분봉 수준(≈1.14e8)으로 정상 범위.

---

## 8. 진단 그래프 해석 (지침 형식: 데이터·모델 → 축 → 진단 목적 → 관찰 → 좋음/나쁨 → 다음 반영)

### 8.A MambaLike__seasonal_diff16__balanced_composite__seed42 (캡처 1·3, 6종)

1. **Total objective**
   - 데이터·모델: 위 케이스의 train/validation 총 손실.
   - 축: x=epoch(1~14), y=총 objective(0.005~0.027).
   - 목적: 수렴 여부와 과적합 점검.
   - 관찰: epoch 1에서 급락 후 0.005 부근 수렴, epoch 10 부근 소폭 반등.
   - 좋음/나쁨: **대체로 좋음**. 수렴은 정상이나 epoch 10 반등이 §4 val 스파이크와 일치.
   - 다음: early-stop patience와 lr 스케줄을 epoch 10 불안정 구간에 맞춰 점검.
2. **Objective components**
   - 데이터·모델: 손실 구성요소 — `huber, direction, variance, correlation, tail, regime, near_zero, mean_bias`.
   - 축: x=epoch, y=각 구성요소 값(0~4).
   - 목적: 어떤 항이 총손실을 지배하는지 분해.
   - 관찰: 초기 variance/correlation 항이 크게 시작해 빠르게 감소, correlation 계열이 1.0 부근 평탄선 유지.
   - 좋음/나쁨: **주의**. correlation 항이 잘 안 떨어지면 방향성 학습이 약하다는 신호.
   - 다음: balanced_composite의 가중치 재조정(특히 direction/correlation) 검토.
3. **Gradient norm**
   - 축: x=epoch, y=grad norm(mean·max, 0~250).
   - 목적: optimization 안정성 점검.
   - 관찰: max가 후반 epoch에서 **약 250까지 스파이크**, mean은 낮게 유지.
   - 좋음/나쁨: **나쁨 신호**. max 스파이크는 gradient clipping 필요성을 시사.
   - 다음: `clip_grad_norm_` 도입 또는 강화, lr warmup 점검.
4. **Return prediction** (캡처 3)
   - 축: x=time index(0~200), y=return(-0.006~+0.006), actual vs predicted.
   - 목적: 수익률 예측의 분산/추종성 점검.
   - 관찰: predicted가 actual보다 **진폭이 작게 압축**되어 평균 회귀하는 경향.
   - 좋음/나쁨: **나쁨(collapse 경향)**. 변동성 과소예측은 금융 시계열의 전형적 붕괴.
   - 다음: variance penalty 완화 또는 분산 보존 손실 항 추가.
5. **Next-candle comparison** (캡처 3)
   - 축: x=index(0~60), y=KRW(1e8 스케일, 1.1375~1.1525), actual candle / predicted proxy / persistence close.
   - 목적: 캔들 단위 예측이 persistence(직전값 복사) 대비 우위인지.
   - 관찰: predicted proxy가 actual을 대체로 따라가나 persistence와 큰 차이 없음.
   - 좋음/나쁨: **경계**. persistence를 못 이기면 예측의 부가가치가 의심됨.
   - 다음: persistence 대비 DA·MASE를 명시 비교(현재 캡처엔 수치 미가시).
6. **Calibration scatter / Pearson=0.097**
   - 축: x=actual return(-0.008~+0.008), y=predicted return.
   - 목적: 예측-실제 정렬도(피어슨 상관) 점검.
   - 관찰: **Pearson=0.097** — 거의 무상관, 산점이 수평 띠로 뭉침.
   - 좋음/나쁨: **명확히 나쁨**. 방향 예측력이 사실상 없음 + 분산 압축 동반.
   - 다음: 이 조합은 후보에서 강등 검토. 입력 feature/horizon 재설계 필요.

### 8.B PatchTSTLike__winsor_025__balanced_composite__seed2026 (캡처 4, 6종)

1. **Total objective**: epoch 1 급락(0.08→0) 후 0 부근 수렴. train/validation 거의 겹침 → 수렴 양호.
2. **Objective components**: 초기 한 항이 ~30까지 치솟았다 급락. 나머지 항은 0 부근. 초기 한 항 지배 후 안정화.
3. **Gradient norm**: max가 후반에 **약 500까지 스파이크**(§5.2 epoch 22 grad=10.5036 등과 정합). mean은 낮음.
   → **8.A보다 더 큰 gradient 불안정**. clipping 우선 적용 대상.
4. **Return prediction**: predicted 진폭이 actual 대비 압축(8.A와 동일 collapse 경향).
5. **Next-candle comparison**: 1e8 스케일(1.142~1.150)에서 actual/predicted proxy/persistence 추종.
6. **Calibration scatter / Pearson=0.094**: **Pearson=0.094** — 8.A와 마찬가지로 거의 무상관.

### 8.C 두 케이스 공통 결론

- 두 대표 케이스 모두 **Calibration Pearson ≈ 0.09**로 수렴해, point 예측의 방향 정렬력이 매우 약하다.
- 두 케이스 모두 **return prediction 분산 압축(collapse 경향)**이 관찰된다.
- 두 케이스 모두 **gradient norm max 스파이크**(250·500)가 있어 optimization stability가 과제다.
- 즉 13번에서 본 point branch의 공통 약점은 (1) 방향 무상관, (2) 분산 붕괴, (3) gradient 스파이크다.

---

## 9. 이번 13번에서 확정적으로 말할 수 있는 사실

1. 13번은 빈 노트북이 아니라 **실제 대규모(1440 케이스) 매트릭스 실험**이 돌고 있었다.
2. point branch 진단 그래프, risk branch 그래프, prediction preview, event preview, fusion 표가
   모두 실제로 생성되고 있었다 → 12번 보고서 구조를 계승·확장한 설계가 구현되어 동작했다.
3. 대표 케이스 2건의 calibration Pearson은 0.094~0.097로, **point 방향 예측력이 거의 없다.**
4. return prediction은 **분산 압축(collapse) 경향**, gradient norm은 **max 스파이크(250·500)**가 관찰된다.
5. 실행은 성능 결론에 도달하기 전, **공유 메모리 부족 + 노트북 저장 실패**라는 자원/저장 장애로 중단됐다.

## 10. 이번 13번에서 말할 수 없는 것 (영구 소실, 재실행 필요)

- 전체 1440 케이스 leaderboard와 top-k
- feature group 평균 / model family 평균 / feature×model heatmap
- final decision / top candidates 요약
- 마지막까지 완료된 case 범위와 그 수치
- 마지막 full traceback 전체

> 위 항목은 "어떤 변수군/모델/risk gate 조합이 최선인가"라는 13번의 핵심 연구 질문에
> 직접 답하는 표들이다. 이번 회차로는 그 결론을 낼 수 없으며, **재실행이 유일한 복원 경로**다.

---

## 11. 재실행 시 반드시 바꿔야 할 것

### 11.1 공유 메모리(shm) 장애 차단
- `num_workers=0` 또는 `1`로 하향(현재 다중 worker가 /dev/shm 고갈 유발).
- `batch_size`를 1024 → 512 등으로 하향, `pin_memory`·큰 collate payload 점검.
- suite를 쪼개 한 번에 도는 case 수 축소.

### 11.2 노트북 output 과대(저장 실패) 차단
- 모든 케이스 그래프를 매번 그리지 않기 — 대표 케이스만 플롯(`case_plots=False` 기본).
- 중간 로그 축약, 큰 표는 top-k·summary만 노트북에 출력.
- 전체 raw 표는 노트북이 아니라 후속 Markdown/CSV 산출물로 분리.

### 11.3 suite 분리 (한 노트북에 1440을 다 찍지 않기)
- `mtf_decomposition` → `algorithm_screen` → `feature_algorithm_matrix` →
  `risk_gate_sensitivity` → `full_resource_top_candidates_only` 순으로 분리.
- "후보 압축 후 심화" 구조로 전환.

### 11.4 모델/최적화 안정성 (이번 캡처가 새로 드러낸 과제)
- gradient norm max 스파이크 대응: `clip_grad_norm_` 도입/강화, lr warmup 점검.
- return prediction 분산 압축 대응: variance penalty 완화 또는 분산 보존 손실 항.
- calibration Pearson ≈0.09 대응: 입력 feature·horizon 재설계, balanced_composite 가중치 재조정.

---

## 12. 관련 산출물 (13번 복구 문서 세트)

- 장애 원인 포렌식: `test/results/13_feature_algorithm_resource_output_recovery_20260627.md`
- 상태 보고서: `test/results/13_feature_algorithm_resource_status_report_20260627.md`
- live recovery dump: `test/results/13_feature_algorithm_resource_live_recovery_dump_20260627.md`
- 부분 파싱 보고서: `test/results/13_feature_algorithm_resource_partial_parsed_report_20260628.md`
- **본 통합본(이 문서)**: `test/results/13_feature_algorithm_resource_full_captured_results_20260628.md`
- 원본 캡처: `test/images/13_output_recovery_20260627/capture_001~004.png`

---

## 13. 한 줄 요약

13번은 1440 케이스 대규모 실험이 실제로 돌았고 point/risk/fusion 출력이 살아 있었으나,
**공유 메모리 부족과 노트북 저장 실패로 성능 결론 전에 중단**됐다. 캡처로 복원 가능한
모든 수치는 본 문서에 전량 옮겼고, **leaderboard·평균표·final summary는 영구 소실되어
재실행이 유일한 복원 경로**다. 캡처가 새로 드러낸 핵심 과제는 calibration 무상관(Pearson≈0.09),
return 분산 붕괴, gradient norm 스파이크다.
