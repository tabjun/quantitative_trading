# 13번 feature-algorithm-resource 부분 파싱 보고서 (캡처 기반)

> 작성 도구: Claude Code (2026-06-28)
> 원본 상태: `test/models/13_feature_algorithm_resource_test.ipynb`는 디스크에 output이
> serialize되지 않았다 (`output_type` 0개, `image/png` 0개, `execution_count` 0개, 40,881 bytes 엔트리포인트 상태).
> 따라서 이 보고서의 source of truth는 **노트북 파일이 아니라 이미 확보된 화면 캡처 4장**이다.
> 캡처 폴더: `test/images/13_output_recovery_20260627/capture_001~004.png`

---

## 0. 이 보고서의 성격과 한계 (먼저 읽기)

이 문서는 13번 "성능 결론 보고서"가 아니다. 13번 실행은 도중에 두 가지 장애로 중단·소실됐다.

1. **실행 중단**: PyTorch `DataLoader` worker가 공유 메모리(`/dev/shm`)를 더 할당하지 못함
   (`RuntimeError: unable to allocate shared memory(shm) ... Resource temporarily unavailable (11)`).
2. **저장 실패**: VSCode가 노트북 출력이 너무 커서 백업/스냅샷 생성에 실패
   (`Error: Notebook too large to backup`). 그래서 일반 저장·다른 이름 저장 모두 불안정해졌고,
   결과가 `.ipynb` JSON으로 내려오지 못했다.

그 결과 **전체 1440 케이스 중 화면에 남아 캡처된 일부 구간만** 복원 가능하다. 사용자가 중간에 Esc로
화면 제어를 중지했기 때문에 leaderboard·feature group 평균·model family 평균·final summary 화면은
캡처되지 못했고, 이들은 디스크 어디에도 없으므로 **복원 불가**다.

이 보고서가 기존 3개 복구 문서와 다른 점은, "캡처가 있다"는 정황 서술을 넘어
**캡처 4장에서 읽어낸 모든 수치를 실제로 표로 파싱**했다는 것이다.

| 구분 | 상태 |
| :--- | :--- |
| 노트북 디스크 output | 없음 (복원 불가) |
| 화면 캡처 4장 | 확보 — 이 보고서에서 전수 파싱 |
| 전체 leaderboard / 평균표 / final summary | 캡처 안 됨 — 복원 불가 |
| 실험 설계·계보 | `test/experiment_specs/13_*` + `test/README.md`로 확인 가능 |

---

## 1. 실행 환경 (캡처에서 확인된 범위)

- 커널: `Python 3.12 (Quant Stat) [quant_uv_py312_20260614_045930]` (캡처 상단 노트북 헤더)
- 실행 위치: 학교 서버 원격 경로 `/home/std_jun99120/personal_ai_project/quantitative_trading/...`
  (캡처 2 traceback에서 확인). 즉 로컬이 아니라 원격 커널에서 대규모 실행 중이었다.
- 노트북 셀 1번 주석: `# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]` (엔트리포인트 미러 셀)
- 화면상 진행 표시: `[case 28/1440]` → **총 계획 케이스 수 1440개**, 캡처 시점은 28번째 부근.

---

## 2. 확정된 실험 축 (캡처 텍스트에서 직접 읽음)

캡처에 실제 문자열로 등장한 조합 축만 확정으로 적는다.

| 축 | 캡처에서 확인된 값 |
| :--- | :--- |
| feature set | `mtf_returns_only` |
| preprocessing | `seasonal_diff16`, `winsor_025` |
| point model (단일/앙상블) | `MambaLike`, `PatchTSTLike`, `TCN` / 조합 `MambaLike + PatchTSTLike`, `TCN + PatchTSTLike` |
| objective | `balanced_composite` |
| risk target | `absolute_move` |
| horizon | `horizon16` |
| seq length | `seq64` |
| seed | `42`, `777`, `2026` |

> 13번 계획서(`test/experiment_specs/13_feature_algorithm_resource_plan_20260625.md`)에는 Linear/PatchTSTLike/
> TCN/ModernTCN/DLinear/NLinear/Transformer/iTransformer/Autoformer/Mamba까지 더 넓은 모델군이 있으나,
> 위 표는 **캡처에 실제로 찍힌 것만** 확정으로 기록한 것이다. 나머지가 돌았는지는 캡처만으로는 알 수 없다.

---

## 3. 파싱한 수치 (1): fusion 전략 요약표 — 캡처 1

캡처 1 상단에 `5 rows x 8 cols` 표가 있고, 가로 스크롤 때문에 **8열 중 앞 4개 수치열, 5행 중 아래 3행**만 보인다.
(행 0,1은 위로, 열 5~8은 오른쪽으로 잘림.) 읽힌 부분만 그대로 옮긴다.

| 행 | 전략 | 열A | 열B | 열C | 열D |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 2 | `risk_gate_only` | -0.9728078171607566 | -0.9729251501572636 | -0.14879707862244232 | 0.5721652766282549 |
| 3 | `point_plus_risk_gate` | -0.9880838073810395 | -0.9880795475592111 | -0.21292424496937234 | 0.25600050932705165 |
| 4 | `no_trade` | 0.0 | 0.0 | 0.0 | 0.0 |

해석 메모 (열 헤더가 잘려 단정은 피함):
- `no_trade` 행이 전부 0.0인 것은 "거래 안 함 = 수익·손실·objective 0" 기준선으로 자연스럽다.
- `risk_gate_only`, `point_plus_risk_gate`의 열A·열B가 -0.97 ~ -0.99로 매우 음수다. 이 열들이 정규화된
  objective/return 계열이라면 이 두 전략이 기준선 대비 강하게 불리하게 나온 구간이라는 뜻이고,
  점예측 단독이 아직 신뢰할 수준이 아니라는 기존 10~12번 결론과 방향이 일치한다.
- 단, **열 헤더가 캡처에 없으므로 이 수치의 정확한 의미(return인지 objective인지 sharpe인지)는 확정 불가**다.

---

## 4. 파싱한 수치 (2): case 28 학습 로그 — 캡처 1

케이스: `[case 28/1440] mtf_returns_only / seasonal_diff16 / MambaLike + PatchTSTLike / seed42 / seq64`

| epoch | train | val | grad | lr |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.027779 | 0.007803 | 0.7681 | 9.96e-04 |
| 2 | 0.006861 | 0.004385 | 1.4563 | 9.83e-04 |
| 3 | 0.005827 | 0.004261 | 2.2450 | 9.62e-04 |
| 4 | 0.005335 | 0.003670 | 2.4762 | 9.33e-04 |
| 5 | 0.004233 | 0.006847 | 2.0772 | 8.97e-04 |
| 6 | 0.005724 | 0.006942 | 2.1944 | 8.54e-04 |
| 7 | 0.005047 | **0.002502** | 2.0370 | 8.04e-04 |
| 8 | 0.004877 | 0.006243 | 2.9462 | 7.50e-04 |
| 9 | 0.004798 | 0.006004 | 5.2536 | 6.91e-04 |
| 10 | 0.007392 | 0.017507 | 6.0345 | 6.29e-04 |
| 11 | 0.006018 | 0.005645 | 4.7306 | 5.65e-04 |
| 12 | 0.005519 | 0.003893 | 2.7140 | 5.00e-04 |
| 13 | 0.003264 | 0.003868 | 2.4865 | 4.35e-04 |
| 14 | 0.003962 | 0.004930 | 2.8744 | 3.71e-04 |

`[early-stop] epoch=14 best_val=0.002502` (best는 epoch 7에서 나옴)

관찰:
- val loss가 epoch 7에 최저(0.002502)를 찍은 뒤 다시 0.006~0.017로 출렁였다. 즉 **단조 수렴이 아니라
  중반 이후 검증손실이 불안정**하다.
- epoch 10에서 val=0.017507로 튀고 같은 구간 grad=6.03으로 커졌다 → 이 케이스도 중반에 한 번 흔들렸다.
- early-stop이 동작해 epoch 14에서 멈춘 점은 안정장치가 작동했다는 뜻이다.

---

## 5. 파싱한 수치 (3): 또 다른 case 학습 로그 — 캡처 3·4

캡처 3 하단과 캡처 4 상단에는 **서로 다른 스케일의 학습 로그 두 개**가 보인다. (연속 구간이 아니라 스크롤된 별도 화면.)

### 5-1. 고(高)스케일 로그 (캡처 3 하단, epoch 1~7) — loss ≈ 0.9 스케일

| epoch | train | val | grad |
| ---: | ---: | ---: | ---: |
| 1 | 0.945247 | 0.882157 | 0.2791 |
| 2 | 0.926420 | 0.861249 | 0.2542 |
| 3 | 0.912067 | 0.904456 | 0.3113 |
| 4 | 0.898672 | 0.939426 | 0.3718 |
| 5 | 0.881307 | 0.883570 | 0.4036 |
| 6 | 0.866108 | 0.927276 | 0.4509 |
| 7 | 0.848609 | 0.933704 | 0.5264 |

→ loss가 ~0.9 스케일에서 천천히 내려간다. case 28(0.005 스케일)과 정규화/objective 기준이 다른 케이스로 보인다.
   val이 train보다 높고 거의 정체 → 이 케이스는 일반화가 약하다.

### 5-2. 저(低)스케일 로그 (캡처 4 상단, epoch 8~24) — loss ≈ 0.002 스케일, lr 동반

| epoch | train | val | grad | lr |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.004546 | 0.002000 | 1.1591 | 7.50e-04 |
| 9 | 0.003754 | 0.002662 | 0.8885 | 6.91e-04 |
| 10 | 0.003674 | 0.003145 | 1.3215 | 6.29e-04 |
| 11 | 0.004252 | 0.003131 | 2.5832 | 5.65e-04 |
| 12 | 0.002965 | 0.001711 | 1.1004 | 5.00e-04 |
| 13 | 0.002959 | 0.001571 | 1.0143 | 4.35e-04 |
| 14 | 0.003349 | 0.003557 | 0.8915 | 3.71e-04 |
| 15 | 0.003022 | 0.001793 | 1.6795 | 3.09e-04 |
| 16 | 0.002698 | 0.001756 | 1.4944 | 2.50e-04 |
| 17 | 0.002561 | 0.001752 | 2.5151 | 1.96e-04 |
| 18 | 0.002096 | 0.001385 | 0.5804 | 1.46e-04 |
| 19 | 0.002003 | 0.001245 | 2.8150 | 1.03e-04 |
| 20 | 0.001939 | 0.001136 | 1.1100 | 6.70e-05 |
| 21 | 0.001954 | 0.001530 | 0.5652 | 3.81e-05 |
| 22 | 0.001900 | 0.001245 | **10.5036** | 1.70e-05 |
| 23 | 0.001866 | 0.001118 | 2.3547 | 4.28e-06 |
| 24 | 0.001869 | 0.001120 | 3.8322 | 0.00e+00 |

관찰:
- val이 0.0020 → 0.0011로 비교적 매끄럽게 내려가 case 28보다 안정적으로 수렴했다.
- 단 **epoch 22에서 grad=10.50으로 급등(grad spike)**. lr이 거의 0인데도 gradient norm이 튀는 건
  특정 배치에서 큰 손실 기울기가 생겼다는 뜻 → 일부 구간 optimization 불안정 신호.

---

## 6. 파싱한 수치 (4): risk event preview 표 — 캡처 2

표 제목: `[PatchTSTLike__mtf_returns_only__winsor_025__absolute_move__horizon16__seed777 event preview]`
(`12 rows x 8 cols`, page 1 of 2. 화면에 보인 10행만 파싱.)

컬럼: `timestamp | prev_close | future_end_close | future_return | event_score | event_label`

| timestamp | prev_close | future_end_close | future_return | event_score |
| :--- | ---: | ---: | ---: | ---: |
| 2026-05-20 00:15:00 | 1.14554e8 | 1.14106e8 | -0.003918 | 0.006165 |
| 2026-05-20 00:30:00 | 1.14251e8 | 1.14061e8 | -0.001664 | 0.003516 |
| 2026-05-20 00:45:00 | 1.14307e8 | 1.14093e8 | -0.001874 | 0.004006 |
| 2026-05-20 01:00:00 | 1.14277e8 | 1.13968e8 | -0.002708 | 0.003744 |
| 2026-05-20 01:15:00 | 1.144e8   | 1.14162e8 | -0.002083 | 0.004819 |
| 2026-05-20 01:30:00 | 1.14515e8 | 1.14342e8 | -0.001512 | 0.005824 |
| 2026-05-20 01:45:00 | 1.14614e8 | 1.14541e8 | -0.000637 | 0.006688 |
| 2026-05-20 02:00:00 | 1.1442e8  | 1.14619e8 | 0.001738  | 0.004994 |
| 2026-05-20 02:15:00 | 1.14136e8 | 1.14536e8 | 0.003498  | 0.004223 |
| 2026-05-20 02:30:00 | 1.14402e8 | 1.14647e8 | 0.002139  | 0.004837 |

관찰:
- `prev_close`가 모두 약 1.14e8 (≈ 1억 1천만 KRW대) → KRW-BTC 15분봉 구간이다.
- `future_return` 절댓값이 대부분 0.001~0.004로 작다. `event_score`(위험 점수)는 0.003~0.007 범위.
- `event_label` 열은 화면에서 잘려 값 미확보.

> 용어: **`absolute_move` 위험 이벤트**란 "향후 horizon 구간 동안 가격이 (방향 무관) 일정 크기 이상
> 움직였는가"를 라벨로 두는 분류 문제다. 점예측(다음 종가 맞히기)과 별개의 분기로, 11번에서 가장 강했던
> 위험 게이트 후보다. 13번은 이 분기를 `horizon16`(향후 16스텝) 기준으로 계속 출력하고 있었다.

---

## 7. 그래프 해석 (지침: 데이터/모델 → x·y축 → 진단 목적 → 관찰 → 좋음/나쁨 → 다음 반영)

### 7-1. case `MambaLike__seasonal_diff16__balanced_composite__seed42` (캡처 1·3)

**(a) Total objective**
- 데이터/모델: 위 케이스의 학습 곡선. x축 = epoch(1~14), y축 = total objective(≈0.005~0.027).
- 목적: 학습/검증 목적함수가 정상적으로 하강하는지.
- 관찰: train은 빠르게 내려가나 validation이 중반(≈epoch10)에 위로 한 번 튄다.
- 좋음/나쁨: **모호**. 수렴은 하지만 검증 곡선이 매끄럽지 않다.
- 다음 반영: warm-up·lr 스케줄을 이 케이스에도 적용해 중반 튐을 줄인다.

**(b) Objective components** (huber / direction / variance / correlation / tail / regime / near_zero / mean_bias)
- x축 = epoch, y축 = 각 보조손실 크기.
- 목적: balanced_composite를 구성하는 보조 항 중 어떤 것이 지배적인지.
- 관찰: 초기 1~2 epoch에 일부 항(초록·회색 계열)이 크게 출발했다가 1 이하로 수렴. `correlation` 항(빨강)은
  거의 1.0 부근에서 평평하다.
- 좋음/나쁨: 보조항이 폭주 없이 가라앉은 건 좋은 신호. 단 correlation 항이 개선되지 않고 평평한 건
  방향성 학습이 잘 안 되고 있다는 뜻일 수 있다.
- 다음 반영: correlation/direction 가중을 재조정해 방향 신호를 더 끌어낼지 검토.

**(c) Gradient norm** (mean / max)
- x축 = epoch, y축 = gradient norm. max가 epoch 10 부근 **≈250까지 스파이크**.
- 목적: 기울기 폭주(exploding gradient) 여부.
- 관찰: mean은 낮게 깔려 있으나 max가 한 번 크게 튄다.
- 좋음/나쁨: **나쁨 신호 후보**. gradient clipping이 있더라도 일부 배치에서 큰 기울기가 발생.
- 다음 반영: `grad_clip_norm` 강화 또는 lr 추가 하향.

**(d) Return prediction** (actual return vs predicted return)
- x축 = test time index(0~200), y축 = return(±0.006).
- 목적: 예측 수익률이 실제 수익률의 변동폭을 따라가는지.
- 관찰: **predicted(주황)의 진폭이 actual(파랑)보다 뚜렷이 작다.** 예측이 0 근처로 압축돼 있다.
- 좋음/나쁨: **나쁨 신호** — 이는 "예측 분산 축소(under-dispersion)", 즉 모델이 변동을 0에 가깝게
  뭉개는 collapse 경향이다. 10~12번에서 반복 관찰된 문제와 동일 계열.
- 다음 반영: variance/anti-collapse 항 가중 강화 검토.

> 용어: **collapse(평탄화/붕괴)**란 모델이 어떤 입력에도 거의 같은(보통 0에 가까운) 값을 내놓는 현상.
> 예측 수익률이 항상 0이면 RMSE는 그럴듯해 보여도 실제로는 아무것도 예측하지 못한 것이다.

**(e) Next-candle comparison** (actual candle / predicted proxy candle / persistence close)
- x축 = 캔들 인덱스(0~60), y축 = KRW(≈1.14e8).
- 목적: 예측 종가 기반 proxy 캔들이 실제 다음 봉과 persistence(직전 종가 유지)와 비교해 얼마나 다른지.
- 관찰: 예측 proxy 캔들이 실제 캔들·persistence와 거의 겹친다.
- 좋음/나쁨: **모호~나쁨** — persistence와 사실상 구분되지 않으면 점예측이 직전가 복사에 가깝다는 신호.
- 다음 반영: persistence 대비 우위(MASE<1)를 정량 확인해야 의미가 생긴다.

> 용어: **persistence(직전값 유지 기준선)**란 "다음 값 = 현재 값"으로 두는 가장 단순한 예측.
> 예: 지금 1억 1,455만 원이면 다음도 1억 1,455만 원이라고 찍는 것. 모델이 이걸 못 이기면 학습 가치가 없다.
> **MASE**는 모델 오차를 이 persistence 오차로 나눈 값으로, 1보다 작아야 persistence를 이긴 것이다.

**(f) Calibration scatter / Pearson=0.097**
- x축 = actual return, y축 = predicted return. 대각선은 완벽 보정선.
- 목적: 예측과 실제의 선형 상관(보정도).
- 관찰: 점들이 대각선 주위로 퍼지지 못하고 가로로 눌려 있으며 **Pearson=0.097(매우 약함)**.
- 좋음/나쁨: **나쁨** — 상관이 0.1 미만이면 점예측이 실제 수익률을 거의 설명하지 못한다.
- 다음 반영: 점예측 단독을 알파로 쓰지 않는다는 12번 결론을 재확인.

### 7-2. case `PatchTSTLike__winsor_025__balanced_composite__seed2026` (캡처 4)
- Gradient norm: max가 epoch 22 부근 **≈500까지 스파이크** (case 28보다 더 큼).
- Calibration scatter: **Pearson=0.094** (역시 매우 약함).
- 해석: 다른 seed·전처리에서도 (1) gradient spike, (2) calibration 0.09 수준이 반복 → 특정 우연이 아니라
  **현재 13번 설정의 공통 경향**으로 보인다.

### 7-3. risk branch 그래프 (캡처 2)
- **Precision-recall curve**: precision이 recall 0 부근 ≈0.6에서 시작해 recall 증가와 함께 ≈0.3
  (점선 baseline)으로 하락. → 위험 이벤트를 적게 골라낼 때는 정밀도가 있으나 많이 잡으려 하면 baseline 수준.
- **Selective risk-coverage**: x축 = 채택 표본 비율(0.2~1.0), y축 = classification error(0.30~0.42).
  채택 비율을 늘릴수록 오류가 증가 → "확신 높은 일부만 거래" 전략이 합리적이라는 신호(좋음).
- **Probability vs realized risk**: 예측 위험확률(0.1~0.9) vs 실제 위험점수(0~0.08) 산점도.
  점들이 넓게 흩어져 보정이 강하지는 않다.

> 용어 정리: **precision**(위험이라 부른 것 중 실제 위험 비율), **recall**(실제 위험 중 잡아낸 비율),
> **selective risk-coverage**(확신 높은 표본만 골라 쓸 때 오류가 줄어드는지 보는 곡선),
> **calibration/Pearson**(예측 확률·값이 실제와 같은 방향·크기로 움직이는 정도, 1에 가까울수록 좋음).

---

## 8. 장애 원인 (요약 — 상세는 기존 복구 메모 참조)

- **실행**: `DataLoader`가 worker→메인으로 배치를 넘길 때 `/dev/shm` 공유메모리 부족으로 중단.
  모델 수식 오류가 아니라 `num_workers`/`batch_size`/출력량이 큰 상태의 자원 한계.
- **저장**: 노트북 출력 누적량이 과대(셀 높이 view state ≈196,677px)해 VSCode 백업 스냅샷 생성 실패.
  이로 인해 일반/다른이름 저장도 같은 큰 payload를 다시 직렬화해야 해서 실패.
- 기존 포렌식: `13_feature_algorithm_resource_output_recovery_20260627.md`,
  `13_feature_algorithm_resource_status_report_20260627.md`,
  `13_feature_algorithm_resource_live_recovery_dump_20260627.md`.
- 로그 경로: `%APPDATA%\Code\logs\20260624T203952\window1\...\7-Jupyter.log`, `...\renderer.log`.

---

## 9. 캡처만으로도 확정되는 연구적 사실

1. 13번은 빈 노트북이 아니라 **실제 대규모 매트릭스(계획 1440 케이스)를 학습 중**이었다.
2. point branch(학습곡선·return prediction·next-candle·calibration)와 risk branch(PR·coverage·calibration)
   **양쪽 진단 출력이 모두 살아 있었다** → 12번 보고 구조를 계승한 확장 실행이 동작했다.
3. 여러 케이스에서 공통으로 (a) **예측 분산 압축(collapse 경향)**, (b) **calibration Pearson ≈0.09(매우 약)**,
   (c) **gradient norm spike(250~500)**가 관찰됐다. → 점예측 단독은 아직 알파로 부적합하다는 10~12번 결론과 일치.
4. risk branch의 selective risk-coverage는 "확신 높은 일부만 거래"가 유리하다는 방향을 보였다(위험 게이트 가치 시사).

## 10. 확정 불가 (복원 안 됨)

- 전체 leaderboard / feature group 평균 / model family 평균 / feature×model heatmap
- final decision·top candidate 요약표 / 마지막까지 완주한 case 수 / 전체 traceback
- 위 fusion 요약표의 잘린 4개 열과 행 0·1
- 이유: 캡처되지 않았고 `.ipynb`에 저장되지 않았기 때문. 재실행 없이는 복원 불가.

---

## 11. 다음 재실행 시 반드시 바꿀 것 (지침: 새 번호로 분리 권장)

> 완료/중단 노트북은 read-only다(CLAUDE.md §2.3). 아래는 13번을 덮어쓰지 말고
> 출력·자원 정책을 고친 **재실행 또는 새 번호 실험**으로 반영한다.

**실행(shared memory)**
1. `num_workers=0` 또는 `1`로 낮춰 `/dev/shm` 의존 제거.
2. `batch_size` 1024 → 필요시 512로 즉시 하향, OOM/shm retry 유지.
3. 서버 컨테이너의 `--shm-size`(또는 `/dev/shm`) 한도 상향 가능 여부 확인.

**저장(notebook output 과대)**
4. case별 상세 플롯을 매번 그리지 말고 **대표 case만** 출력(`case_plots` 기본 off).
5. leaderboard/요약표는 노트북 최하단에 1회만, 전체 raw table은 후속 Markdown/CSV로 분리.
6. full_resource를 한 노트북에 몰지 말고 `mtf_decomposition → algorithm_screen →
   feature_algorithm_matrix → risk_gate_sensitivity → full_resource_top_candidates_only`로 쪼갠다.

**모델/최적화(캡처에서 드러난 경향)**
7. return prediction 압축(collapse) 대응: variance/anti-collapse 항 가중 점검.
8. gradient spike 대응: `grad_clip_norm` 강화 또는 lr 추가 하향.
9. calibration ≈0.09: 점예측 단독을 알파로 쓰지 않고 risk gate 결합(12번 방향) 유지.

---

## 12. 핵심 한 줄

13번 디스크 노트북은 출력이 비어 복원 불가지만, **확보된 캡처 4장에서 실제 수치를 전수 파싱한 결과**
13번은 1440 케이스 중 28번 부근까지 point/risk 양 분기를 정상 출력하며 돌다가 shared memory·저장 과대로
중단됐고, 그 구간 데이터는 기존 10~12번의 collapse·약한 calibration·gradient spike 경향을 그대로 재확인시킨다.
전체 성능 결론은 출력·자원 정책을 고쳐 재실행해야 얻을 수 있다.
