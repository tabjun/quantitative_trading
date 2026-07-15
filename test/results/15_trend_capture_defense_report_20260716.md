# 15번 보고서 — 변동·추세 포착 최적화와 방어 gate 융합

작성일: 2026-07-16 · 브랜치: `stock` · 실험: `test/models/15_trend_capture_defense_test.py`

> 이 문서는 독립 보고서다. 이전 보고서를 읽지 않았다고 가정하고, 결과 해석에 필요한 용어와
> 지표를 모두 다시 풀어 쓴다. suite별 전체 케이스 표와 실행 인자 원본은
> `test/results/15_trend_capture_defense_20260716/` 아래 `*_raw.md`/`*_leaderboard.csv`에 있다.

---

## 0. 한 줄 요약

예측 target을 "다음 15분 수익률"에서 **"향후 4시간(16봉) 누적수익률(추세)"**로 바꾸면 예측이
0-예측 기준선에 크게 다가서고(평균 오차비 2.52 → 1.27), 큰 변동 구간의 방향 정확도가 단순
추세지속 가정(0.465)보다 높은 0.51~0.53까지 올라온다 — 즉 **"변동을 살린 예측"이 처음으로
평탄화·폭주 없이 확보**됐다. 다만 그 신호는 아직 약해서(상관 0.03~0.07) 거래비용을 이기지
못하며, risk gate는 이번에도 모든 설정에서 최대낙폭(MDD)을 일관되게 줄였다(평균 -0.33 → -0.18).
결론: 추세 포착 축은 유효한 방향이되 신호 강화가 필요하고, gate는 그 위에 얹는 방어층으로 확정.

---

## 1. 왜 방향을 바꿨나 (문제 재정의)

- 8~14번 내내 "다음 15분 수익률" 점예측은 persistence(직전값 복사)를 못 이겼다. 15분 수익률은
  신호 대 잡음비가 사실상 0에 가까운 target이라, 학습이 ① 분산 폭주(예측이 실제보다 수천 배
  요동) 아니면 ② 0 근처 평탄화(예측 그래프가 변동 없이 눌림) 사이를 오갔다.
- 13번(부분 결과)의 예측 그래프가 변동이 없어 보인 것은 "하방 방어 우선" 셋업과 평탄화 모드가
  결합해 강화된 결과다. 방어만 할 거면 모델링이 아니라 로직 분기로 충분하다 — 연구 목적은
  **변동성이 큰 추세 자체를 잘 예측**하는 비정상 시계열 최적화·변수 구성을 찾는 것이다.
- 그래서 15번은 target을 h-step 누적수익률로 바꾸고(11번 risk branch가 이벤트 분류용으로만
  쓰던 구조를 회귀로 전환), objective·정규화·전처리·변수 구성을 다시 훑은 뒤, 마지막에
  risk gate를 융합해 방어 효과를 함께 봤다.

## 2. 실행 및 분석 환경

- 학교 서버 컨테이너, NVIDIA RTX 4090 24GB, CUDA 12.6, Python 3.12.13, torch 2.10.0+cu126
- **데이터 재수집**: 2026-07-14 서버 홈 초기화로 기존 DuckDB가 소실되어, Upbit 공개 API에서
  KRW-BTC 15분봉 3년치를 재수집했다(`pipelines/rebuild_price_mart.py`). 104,865행,
  2023-07-17 ~ 2026-07-16, 결측 0. **12~14번과 데이터 구간이 다르므로 절대값 비교는 하지 않는다.**
- 15분 로그수익률 std 0.00239, 첨도 107(극단적 heavy-tail — 대부분 조용하다가 가끔 폭발).
- 윈도우: seq_len 64(16시간 문맥), stride 4, 최근 16,000윈도우(약 22개월). 시간순
  70/15/15 분할 — 학습 2024-09~2025-12, **테스트 2026-04-06~2026-07-15**(buy&hold -7.7%,
  고점 대비 -20.6%의 하락·급변장이라 방어 검증에 적합한 구간).
- 실행 방식이 이번부터 노트북 셀이 아니라 **헤드리스 `.py` 드라이버**다(13번 notebook 저장
  실패·출력 용량 문제 교훈). 모든 결과는 실행 즉시 `test/results`/`test/images`에 md/csv/png로
  저장된다. `.ipynb`는 커밋 정책용 미러다.

## 3. 지표 용어 풀이 (h-step 추세 기준)

- **h-step 누적수익률**: 지금부터 h개 봉(15분×h) 동안의 로그수익률 합. h=16이면 "향후 4시간
  동안 얼마나 오르내리나". 이것이 이번 예측 대상이다.
- **mae_zero_ratio** = 모델 MAE ÷ "0 예측" MAE. 0 예측은 "가격은 랜덤워크라 못 맞힌다"는
  기준선. **1 미만이면 랜덤워크 가정을 이긴 것.** (1-step에서의 persistence 대비 copy_risk와
  같은 취지의 h-step 버전)
- **mase_momentum** = 모델 MAE ÷ "직전 h봉 수익률이 그대로 이어진다(추세지속)" MAE.
  1 미만이면 단순 momentum 가정보다 정확.
- **trend_corr**: 예측 vs 실제 h-step 수익률의 Pearson 상관. 0이면 무관, 1이면 완벽.
  0.05 수준이면 "약하지만 존재하는 신호".
- **large_move_da**: |실제 수익률|이 상위 25%인 **큰 변동 구간만**의 방향(상승/하락) 정확도.
  이번 연구 질문(큰 추세를 잡는가)에 가장 직접적인 지표. 비교 기준으로 momentum naive의 같은
  지표(이번 테스트 구간 0.465~0.527)를 함께 본다.
- **variance_ratio**: 예측분산 ÷ 실제분산. 1 근처가 건강. ≪0.1이면 평탄화(예측이 죽음),
  ≫10이면 폭주. 0.05~20 밖은 랭킹에서 감점 처리해 "평탄화로 오차만 줄인" 케이스를 걸렀다.
- **MDD(최대낙폭)**: 자산곡선이 고점 대비 가장 많이 빠진 비율(0에 가까울수록 방어 우수).
- **risk gate**: 향후 4시간 급변 확률을 별도 분류기로 추정해, 위험 확률이 높으면 신규 진입을
  막는 안전장치(11번 계보, `absolute_move` 이벤트). gate 분위수(0.45/0.55/0.65)가 낮을수록
  더 공격적으로 차단한다.

## 4. T1 — target horizon 스크리닝 (24케이스)

h ∈ {1, 4, 16, 64} × 비폭주 모델 6종(PatchTSTLike, DLinearLike, NLinearLike, TCN,
ModernTCNLike, ITransformerLike; Linear는 14번에서 폭주 확정으로 배제).

| horizon | mae_zero_ratio 평균 | trend_corr 평균 | large_move_da 평균 | variance_ratio 평균 |
|---|---|---|---|---|
| 1 (15분) | 2.522 | -0.013 | 0.489 | 6.61 |
| 4 (1시간) | 1.646 | 0.008 | 0.518 | 1.93 |
| **16 (4시간)** | **1.267** | 0.005 | 0.503 | 0.53 |
| 64 (16시간) | 1.182 | -0.013 | 0.479 | 0.34 |

- **horizon을 늘릴수록 예측 오차가 기준선에 수렴**한다(2.52 → 1.18). "15분 1-step은 노이즈"
  가설과 일치. 다만 어떤 h에서도 평균적으로 1 미만(기준선 돌파)은 아직 아니다.
- h=64는 오차비는 낮지만 방향 지표가 무너지고(large_move_da 0.479), 유효 표본도 줄어든다.
  h=16이 오차·방향·변동 보존의 균형점이고 11번 risk 계보(4시간)와 정렬되므로 **h=16 채택**.
- 모델별: TCN·DLinear는 평탄화(variance_ratio 0.002~0.06), NLinear는 단기 h에서 폭주(32.7),
  **ITransformerLike만 h=16/64에서 "건강한 분산 + 양의 상관"**(0.22~0.37, corr ≈ 0.04)을 유지.

### Fig T1-1. horizon별 mase/trend_corr/variance_ratio 막대
(`test/images/15_trend_capture_defense_20260716/t1_horizon/fig_horizon_*.png`)

- 데이터/대상: 24케이스를 horizon별 평균한 지표들.
- x축: 지표값, y축: horizon(4개).
- 진단 목적: target 길이가 예측 가능성을 바꾸는지(15번의 1차 연구 질문).
- 관찰: mase_momentum·mae_zero_ratio가 h와 함께 단조 감소, variance_ratio는 1-step의 폭주에서
  h=16의 0.5 수준으로 안정화.
- 좋음/나쁨: 좋은 신호 — target 교체가 실패 모드(폭주·평탄화) 자체를 완화했다.
- 다음 반영: 이후 suite는 h=16 고정.

## 5. T2 — objective 스크리닝 (24케이스, h=16)

objective 8종(huber 단독, 방향/분산/상관/tail/regime 보조항 합성 등) × 상위 3모델.

- 1위 **ITransformerLike + 순수 huber**: trend_corr **0.071**, large_move_da **0.523**,
  variance_ratio 0.356(건강). 2위 tail_huber(0.0685/0.5217)로 사실상 동률.
- 8~10번에서 1-step collapse 대응용으로 설계했던 합성 objective들(balanced_composite 등)은
  h=16 추세 target에서는 **huber 단독보다 낫지 않았다**. anti_collapse_v2는 오히려 분산 폭주
  (8~14배)를 유발 — 1-step용 처방을 추세 target에 그대로 이식하면 역효과라는 정직한 결과.
- ModernTCNLike는 objective와 무관하게 동일한 평탄화 해로 수렴(8개 objective 지표가 전부 동일)
  → 후보 탈락. **objective=huber, 모델=ITransformerLike(주) / PatchTSTLike(보조) 채택.**

### Fig T2-1. best 케이스 예측 vs 실제 overlay
(`test/images/15_trend_capture_defense_20260716/t2_objective/fig_best_overlay.png`)

- 데이터/대상: ITransformerLike·h=16·huber·mtf 변수셋(seed42)의 테스트 마지막 400 결정점.
- x축: 결정 시각(2026-06-29~07-15), y축: 향후 4시간 누적 로그수익률. 검정=실제, 빨강=모델 예측,
  하늘색=momentum naive. 하단: 실제 vs 예측 산점도(점선=완벽 예측선).
- 진단 목적: "예측이 변동을 살렸는가"를 눈으로 직접 확인(13번 평탄화 그래프의 반례 검증).
- 관찰: 예측선이 0에 눌려 있지 않고 ±0.01 범위에서 실제와 유사한 스케일로 요동한다. 다만
  산점도의 기울기가 완만해 큰 극단값(±0.02~0.03)은 과소 예측한다.
- 좋음/나쁨: 절반 좋음 — 변동은 살아났지만(평탄화 탈출) 진폭 포착은 부분적.
- 다음 반영: 큰 변동 진폭을 더 키우는 손실 가중(tail 강화)·데이터 확장은 16번 후속 축.

## 6. T3 — 비정상성 처리(정규화×전처리) 스크리닝 (18케이스)

정규화 {window_standard(RevIN성 instance norm), window_robust, asinh_revin} ×
전처리 {seasonal_diff16, winsor_025, none} × 상위 2모델, objective=huber.

- 1위: **window_standard + seasonal_diff16 + ITransformerLike** (trend_corr 0.071). 상위 3개가
  모두 seasonal_diff16 — **16봉(4시간) 계절 차분이 추세 신호 보존에 핵심**.
- winsorize(극단값 절단)와 무전처리는 상관을 0 이하로 죽였다. 큰 변동을 잡으려는 목적에서
  극단값을 자르는 전처리가 해롭다는, 방향성에 부합하는 결과.
- 정규화 간 차이는 부차적(window_standard ≥ asinh_revin > window_robust).

## 7. T4 — multi-timeframe 변수 분해/확장 (16케이스, seed 2개)

12·14번 1위 변수셋 `coin_multitimeframe_structure`(mtf)를 4개 하위 블록으로 분해하고
3개 확장을 비교. 모델 ITransformerLike, huber, seed {42, 7}.

| feature set | trend_corr 평균 | large_move_da 평균 | mase_momentum 평균 |
|---|---|---|---|
| **mtf full (유지)** | **0.043** | 0.513 | 0.783 |
| mtf_trend (추세 블록만) | **0.048** | 0.500 | **0.736** |
| mtf_plus_momentum (확장) | 0.032 | **0.518** | 0.799 |
| mtf_plus_shock / volregime | 0.027 / 0.043 | 0.513 / 0.501 | 0.798 / 0.771 |
| mtf_returns (수익률 블록만) | -0.002 | 0.478 | 0.760 |
| mtf_volume_range (거래량 블록만) | -0.014 | 0.477 | 0.775 |

- **추세 신호를 끄는 핵심 블록은 `mtf_trend`**(price_z_192, trend_strength_64/192): 4개 변수만으로
  full셋의 상관을 재현하고 momentum naive 대비 오차는 가장 낮다. 수익률·거래량 블록 단독은
  신호가 없다(corr ≈ 0). → 데이터마트 승격 시 이 블록이 최소 핵심.
- 확장은 momentum 결합이 큰 변동 방향 정확도를 소폭 올리는 정도(0.518)로, full셋 대비 결정적
  우위는 없다.
- **seed 간 편차가 크다**(mtf full: corr 0.071(s42) vs 0.015(s7)). 신호가 약해 seed 우연에
  민감하다는 한계를 정직하게 기록한다.

### Fig T4-1. feature set별 trend_corr / large_move_da 막대
(`test/images/15_trend_capture_defense_20260716/t4_feature/fig_feature_set_*.png`)

- 데이터/대상: 16케이스를 feature set별 평균(seed 2개 평균).
- x축: 지표값, y축: 변수셋 8종.
- 진단 목적: mtf 방어 우위(12·14)가 "어느 블록" 덕분인지 추세 관점에서 분해.
- 관찰: trend 블록 > full ≈ 확장들 > volatility > returns ≈ volume_range.
- 좋음/나쁨: 좋음 — 변수셋 내부의 기여가 특정됐고, 이는 12·14의 "mtf가 왜 이기는가"에 대한
  첫 번째 기계적 설명(추세·위치 변수가 신호원)이다.
- 다음 반영: 데이터마트 스키마 1순위 = mtf_trend 블록 + full mtf 유지.

## 8. T5 — 추세 정책 × risk gate 융합 (36케이스: 변수셋 3 × seed 3 × gate 4)

추세 예측(ITransformerLike, huber, h=16)으로 long/flat 정책(진입 임계 20bps, 비용 14bps,
**비중복 4시간 간격 의사결정**으로 h-step 겹침 이중계산 제거)을 만들고, 같은 윈도우에서 학습한
급변 분류기(absolute_move, event quantile 0.70, AP lift 1.09~1.45)로 gate를 얹었다.
테스트 구간은 buy&hold -7.7%, 고점 대비 -20.6%의 하락장.

| gate | 누적수익 평균 | MDD 평균 | active_share 평균 |
|---|---|---|---|
| 없음 | -0.309 | -0.331 | 0.393 |
| 0.65 | -0.202 | -0.231 | 0.236 |
| 0.55 | -0.182 | -0.205 | 0.197 |
| **0.45 (공격적)** | **-0.156** | **-0.176** | 0.163 |

- **gate가 공격적일수록 MDD가 단조 감소**(-0.331 → -0.176, 47% 개선), 36/36 케이스 모두 활동
  하한(active_share ≥ 0.05, trade_count ≥ 5) 통과 — "미거래 둔갑"이 아니라 실제 방어다.
- 그러나 **어떤 조합도 절대 수익은 음수**다. 추세 신호(corr 0.03~0.07)가 거래비용(왕복
  14bps × 100~300회)과 하락장을 이기기엔 아직 약하다. gate는 손실을 절반으로 줄일 뿐 흑자로
  뒤집지는 못한다 — 이것이 이번 회차의 가장 중요한 정직한 한계다.
- 변수셋 중에서는 mtf_trend + gate 0.45~0.55가 최소 손실(-0.04~-0.08, seed7)이었으나 seed 간
  편차가 커 확정하지 않는다.

### Fig T5-1. gate 공격성별 방어-수익 trade-off 산점도
(`test/images/15_trend_capture_defense_20260716/t5_gate_fusion/fig_tradeoff.png`)

- 데이터/대상: 36케이스(활동 하한 통과분)의 (MDD, 누적수익), gate별 색.
- x축: MDD(우측=방어 우수), y축: 누적수익(위=수익 우수).
- 진단 목적: gate 강도가 방어-수익 곡선의 어디로 옮기는지.
- 관찰: gate=none(회색)이 좌하단(낙폭·손실 최대), gate가 세질수록 우상단으로 이동. 색 군집이
  겹치지 않고 정렬 — 우연이 아니라 구조적 효과.
- 좋음/나쁨: 방어 관점 좋음. 단 y>0 영역에 점이 없음 — 수익 전환은 미해결.
- 다음 반영: gate는 유지하되, 수익 전환은 신호 강화(아래 16번 후보 축)로 풀어야 한다.

### Fig T5-2. 대표 equity curve
(`test/images/15_trend_capture_defense_20260716/t5_gate_fusion/fig_equity_*.png`)

- 데이터/대상: 변수셋×seed별 테스트 구간 자산곡선(gate 4종 + buy&hold 점선).
- x축: 시각(2026-04~07), y축: 자산배수.
- 진단 목적: gate가 하락 구간에서 실제로 노출을 줄이는지 시계열로 확인.
- 관찰: 5~6월 급락 구간에서 gate 곡선들이 무gate 곡선보다 완만하게 빠지고, 회복 구간 참여는
  줄어든다(전형적 방어 trade-off).
- 좋음/나쁨: 방어층으로서 좋음.

## 9. 종합 결론

1. **target 교체가 최적화 문제를 실제로 바꿨다**: 15분 1-step에서 불가능했던 "평탄화·폭주 없는
   예측"이 h=16 추세 target에서 처음 확보됐다(ITransformerLike + huber + seasonal_diff16 +
   window_standard). 비정상 시계열 대응은 합성 objective보다 **target 설계 + 계절 차분 +
   instance norm** 조합이 유효했다.
2. **변수 구성**: mtf 우위의 원천은 추세·위치 블록(`mtf_trend`)이다. 데이터마트 승격 1순위
   근거가 "방어 MDD 1위"(12·14)에서 "추세 신호원"(15)으로 한 층 더 구체화됐다.
3. **방어 융합**: risk gate는 추세 정책 위에서도 MDD를 47% 줄였다(36/36 일관). 방어는 로직이
   아니라 학습된 gate로 달성 가능함이 재확인됐고, Defense-First 자산은 유지·계승한다.
4. **한계(다음 회차의 출발점)**: 추세 상관 0.03~0.07은 거래비용을 못 이긴다. seed 민감성이
   크다. 단일 자산(BTC)·단일 구간 테스트다.

## 10. 다음 단계 (16번 후보 축)

- **신호 강화**: 학습 데이터 확장(stride 1 전체 3년), hidden/depth 소폭 확장 + Double Descent
  체크, 큰 변동 가중 손실(tail_huber 가중 상향) 재시험, 예측을 분위수(quantile)로 바꿔 진폭
  과소예측 보정.
- **비용 인식 정책**: 진입 임계를 예측 불확실성(예: seed ensemble 분산)으로 조절해 거래 수 절감.
- **다자산 검증**: KRW-ETH 등으로 mtf_trend 블록의 이식성 확인(cross-market mart 확장과 연계).
- **gate 고도화**: downside 전용 이벤트(absolute_move 대신)와 gate-추세 결합 방식(차단 대신
  포지션 크기 축소) 비교.

## 11. 개발·실행 기록

- 서버 홈 초기화로 소실된 DuckDB를 `pipelines/rebuild_price_mart.py`(신규, 재사용 가능)로 재수집.
- 드라이버는 engine/ 교정본을 그대로 재사용(원본 8~14 노트북 미수정, read-only 유지). 윈도우
  캐시로 suite 내 중복 전처리 제거, OOM 자동 배치 축소, DataLoader num_workers=0.
- suite 5개 순차 실행, 실패 케이스 0. suite별 raw md/csv/png 자동 저장(13번 유실 사고 재발 방지).
- T5는 matplotlib CJK 폰트 설정 후 동일 seed로 재실행해 그림 한글 깨짐만 교정(수치 동일).
