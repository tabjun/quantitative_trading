# 15번 계획서(개정) — 변동·추세 포착 최적화 + 방어 gate 융합

작성일: 2026-07-16 · 브랜치: `stock` · 토대: 공용 엔진 `engine/`(12 fusion 결함 ①②③ 교정본)

> 2026-06-30자 `15_multitimeframe_defense_redesign_plan_20260630.md`를 **대체**하는 개정 계획서다.
> 기존 계획서는 삭제하지 않고 이력으로 남긴다. 개정 사유는 사용자 방향 재정의다:
> "하방 방어에만 몰두해 예측 자체가 평탄해지는(변동 없는) 모델은 목적이 아니다.
> **변동성이 큰 추세 자체를 잘 예측**하도록 비정상 시계열 최적화 방식과 변수 구성을 탐색하고,
> risk gate는 버리지 말고 함께 버무린다."

---

## 1. 문제 재정의 — 왜 10~14번의 예측이 평탄했나

10~14번에서 반복 확인된 사실:

- 점예측(다음 **15분** 수익률)은 모델·전처리·objective를 바꿔도 persistence(직전값)를 못 이겼다
  (copy_risk 17~91). 15분 수익률은 신호 대 잡음비가 사실상 0에 가까운 target이다.
- 그 결과 학습은 두 실패 모드 사이를 오갔다: ① 분산 폭주(Linear, variance_ratio 수천),
  ② 0 근처 평탄화(방어 관점에선 안전해 보이지만 예측으로서는 무의미).
- 13번(부분 결과)의 예측 그래프가 10/11/12 대비 변동이 없어 보인 것은 이 평탄화 모드가
  "방어 우선" 셋업과 결합해 강화된 결과로 해석된다.

**가설**: target 자체가 문제다. 15분 1-step 수익률 대신 **h-step 누적수익률(추세)**을 target으로
잡으면 신호 대 잡음비가 올라가 "변동을 살린 예측"이 가능해질 수 있다. 11번 risk branch가
이미 같은 구조(향후 16봉 누적경로)를 이벤트 분류로 쓰고 있었으나, **회귀(추세 크기·방향)**로는
한 번도 시도하지 않았다 — 이것이 15번의 공백이다.

## 2. 연구 질문

1. **Target/horizon**: 다음 1봉(15분)이 아니라 4봉(1h)/16봉(4h)/64봉(16h) 누적수익률을 예측하면
   naive 기준선(0 예측, momentum 지속)을 이길 수 있는가? 변동(분산)을 살리면서?
2. **Objective(비정상 시계열 최적화)**: Huber 단독 대비 방향/분산/상관/tail/regime 보조항이
   추세 포착(상관·큰 변동 방향 정확도)을 실제로 개선하는가?
3. **정규화/전처리(비정상성 처리)**: window_standard(RevIN성 instance norm), robust, asinh 계열과
   seasonal_diff/winsorize 조합 중 무엇이 추세 신호를 보존하는가?
4. **변수 구성**: 12·14 1위 `coin_multitimeframe_structure`의 내부 블록(수익률/변동성/추세/거래량)
   중 어느 블록이 추세 포착을 끄는가? momentum·shock·vol-regime 확장이 더해지면 나아지는가?
5. **방어 융합**: 추세 신호로 만든 포지션에 11번 계열 risk gate를 얹으면, 추세 수익을 크게
   깎지 않으면서 MDD를 줄이는가? (gate 공격성 {0.45, 0.55, 0.65} 민감도)

## 3. 실험 설계

### 공통 backbone

- 데이터: DuckDB `data/upbit_data.db` `btc_15m_advance` (2023-07 ~ 2026-07 재수집본, 104,865행)
- 윈도우: `engine.windows.build_risk_windows` 재사용 — h-step 누적수익률 `future_return`을
  회귀 target으로 사용. point·risk가 **같은 윈도우/decision_timestamp**를 공유하므로 정렬(fix①) 자명.
- 학습: `engine.point.train_model`(objective 구성 재사용), OOM 자동 배치 축소, seed 고정.
- 평가(전 suite 공통, test split):
  - `mae_zero_ratio` = MAE / MAE(0 예측): **1 미만이면 random-walk 기준선을 이긴 것**
  - `mase_momentum` = MAE / MAE(직전 h봉 수익률 지속): 1 미만이면 momentum naive보다 우수
  - `trend_corr` = 예측 vs 실제 h-step 수익률 Pearson 상관
  - `direction_accuracy`(전체) / `large_move_da`(|실제|가 상위 25%인 큰 변동 구간의 방향 정확도)
  - `variance_ratio`(예측분산/실제분산, 1 근처 건강 / ≪0.1 평탄화 / ≫10 폭주), `near_zero_share`
  - KRW 원스케일 MAE와 persistence(변화 없음) 대비 비율
- 랭킹: 건강 gate(0.05 ≤ variance_ratio ≤ 20) 통과 케이스 안에서 `large_move_da`·`trend_corr`
  우선, `mase_momentum` 보조. 평탄화로 지표만 좋아 보이는 케이스를 걸러낸다.

### 단계별 suite (13번 crash 교훈: 헤드리스 .py, per-case 그래프 off, 단계 분리)

| suite | 축 | 격자 | 고정값 |
|---|---|---|---|
| T1 `horizon_screen` | target horizon × 모델 | h {1,4,16,64} × 모델 {PatchTSTLike, DLinearLike, NLinearLike, TCN, ModernTCNLike, ITransformerLike} = 24 | mtf full, seasonal_diff16, window_standard, balanced_composite, seed42 |
| T2 `objective_screen` | objective × 모델 | objective 8종 × T1 상위 3모델 = 24 | T1 최적 horizon |
| T3 `nonstationarity_screen` | 정규화 × 전처리 | norm {window_standard, window_robust, asinh_revin} × prep {seasonal_diff16, winsor_025, none} × 상위 2모델 = 18 | T2 최적 objective |
| T4 `feature_decomposition` | 변수 블록 | mtf 하위 4블록 + mtf full + 확장 3종 = 8셋 × 최적 모델 × seed {42,7} = 16 | T3 최적 norm/prep |
| T5 `gate_fusion` | gate 민감도 | 상위 변수셋 3 × gate {없음, 0.45, 0.55, 0.65} × seed {42,7,123} = 36 | risk: absolute_move, h=16 (11번 계보) |

- Linear는 14번에서 분산 폭주가 모델 고유 결함으로 확정됐으므로 배제한다.
- T5 정책 평가: **비중복(h봉 간격) 의사결정**으로 long/flat 정책을 구성하고 거래비용 14bps를
  반영해 누적수익·MDD·active_share·trade_count를 gate 유/무로 비교한다. 활동 하한
  (active_share ≥ 0.05, trade_count ≥ 5) 미달은 우승 후보에서 분리한다(fix③ 유지).

### 산출물 (전 결과 raw 저장 — .py 헤드리스 실행이므로)

- 드라이버: `test/models/15_trend_capture_defense_test.py` (+동명 `.ipynb` 미러)
- raw 결과: `test/results/15_trend_capture_defense_20260716/<suite>_raw.md`
  (환경, 격자, 전 케이스 leaderboard, naive 기준선, 학습 로그 요약) + 동명 `.csv`
- 이미지: `test/images/15_trend_capture_defense_20260716/<suite>/` — suite별 비교 그래프,
  대표(최상/최하) 케이스의 예측 vs 실제 overlay·scatter, T5 equity curve
- 최종 보고서: `test/results/15_trend_capture_defense_report_20260716.md` (독립 문서)

## 4. 성공 기준

- h-step 추세 target에서 `mae_zero_ratio < 1` 또는 `mase_momentum < 1`이면서
  variance_ratio가 0.1~10인(평탄화도 폭주도 아닌) 케이스가 존재하는가 → 있으면 "변동을 살린
  예측"이 처음으로 확보된 것이고, 없으면 그 사실 자체가 horizon별로 정직하게 기록돼야 한다.
- 큰 변동 구간(large_move_da)이 0.5를 유의하게 넘는 조합 특정.
- gate 융합이 추세 수익을 크게 깎지 않으면서 MDD를 줄이는 설정 (trade-off 곡선) 확보.
- multi-timeframe 내부 블록의 기여 순위 확정 → 데이터마트 승격 변수셋 근거 강화.

## 5. 리스크와 방어

- **재수집 데이터가 12~14와 구간이 다름**(서버 초기화로 DB 소실, 2023-07~2026-07 새 3년):
  과거 결과와 절대값 비교는 하지 않고, 추세는 "기준선 대비 비율 지표"로만 해석한다.
- 큰 h에서는 인접 윈도우 target이 겹쳐 자기상관이 커진다 → 평가 지표는 그대로 두되 T5 정책
  평가는 비중복 의사결정으로 수행해 이중계산을 피한다.
- suite 간 최적값 승계는 드라이버 인자로 명시(로그에 남김)해 silent default를 금지한다.

---

## 6. 개정 2 (2026-07-16 오후) — 판독 가이드 + 신호 강화 축 + LLM 융합 전환 조건

T1~T5 1차 실행 후 사용자 피드백 2건을 반영한다.

### 6.1 판독 가이드 (각 그래프·축이 무엇인지 결과 단에 자동 기록)

- 문제: T1~T5 결과가 축 이름(horizon/objective/nonstationarity)만 있고 "무엇을 왜 비교하는
  실험인지"와 지표 의미가 결과 파일에 없어 숫자만 보고는 해석 불가였다.
- 조치: 드라이버에 `SUITE_INTENT`(suite별 고정/변화/질문/읽는 법)와 `METRIC_GLOSSARY`(지표 사전)를
  상수로 박고, `save_raw_report`가 매 raw md 상단에 자동 출력한다. 이후 실험도 재사용한다.

### 6.2 T6~T8 — 예측 신호 강화 (사용자 선택: "예측축 더 강화")

T1~T5 실측은 냉정하다: `trend_corr` 최고 0.071, `r2` 전부 음수, `large_move_da` 0.51~0.52로
"예측이 죽지도 폭주도 안 하는 상태"까지는 왔으나 실제 예측력은 미미하다. 예측축을 더 밀어
신호가 오르는지 세 갈래로 확인한다.

- **T6 signal_boost**: 우승 구성을 여러 seed로 학습해 seed 평균 앙상블이 단일 seed 대비
  trend_corr/large_move_da를 올리는지. 데이터 규모 효과는 `--max-windows/--stride`로 회차 분리.
- **T7 amplitude**: tail 가중·분산 보존 objective와 전처리로 진폭 과소예측(variance_ratio≈0.36)을
  1 근처로 끌어올리면서 trend_corr를 유지할 수 있는지.
- **T8 multiasset**: 우승 구성을 ETH/XRP/SOL에도 적용해 신호가 BTC 전용인지 일반적인지.
  (선행: `pipelines/rebuild_price_mart.py --ticker KRW-ETH --table eth_15m_advance` 등으로 종목별
  테이블 수집 후 `--ticker-tables` 매핑. T6/T7에서 신호가 실제로 올라야 T8이 의미 있다.)

또한 사후 감사(`test/results/15_direction_audit_10_to_14_20260716.md`)에서 14번의 "Linear 폭주
확정"이 규모 1/20 재실행 아티팩트일 가능성이 확인됐으므로, **T6/T7에서 Linear를 CLI로 복권**해
12번 규모(40k행급)에서 재검증한다.

### 6.3 LLM 융합 전환 조건 (사용자 방향 경고 반영)

> 사용자 경고(2026-07-16): "이 방향(예측 최적화)만 계속 파면 이 분석만 하게 된다. 이것과 다른
> 분석도 해서 결국 **LLM에 태우는 게 목적**이다. 언제까지 이것만 할 수 없다."

이 프로젝트의 원래 목적(process.md Phase 2~4)은 **과거 유사국면(DTW) + 텍스트 컨텍스트(감성/
이벤트) + 예측/위험 신호를 LLM에 태워 상황 복기·판단을 생성**하는 것이다. 예측은 그 재료 중
하나일 뿐이다. 따라서 예측축은 무한 반복하지 않고 아래 조건에서 종료·전환한다.

- **전환 트리거(둘 중 하나면 예측축 종료 → 17번 LLM 융합 착수)**:
  1. T6~T8을 다 돌려도 `trend_corr`가 어떤 우승 구성에서도 0.15를 못 넘고 `large_move_da`가
     0.53을 못 넘는다(= 신호 강화 실패, 예측만으로는 한계).
  2. 또는 T6/T7에서 신호가 유의하게 올라 "쓸 만한 예측 신호 1세트"가 확정된다(= 재료 확보 완료).
- 두 경우 모두 다음 번호(17)는 예측 튜닝이 아니라 **LLM 융합 파이프라인**이다:
  예측값·risk gate·DTW 과거 유사국면·텍스트 컨텍스트를 프롬프트로 묶어 LLM이 상황 판단을
  내리게 하고, 그 판단의 유용성을 평가한다. (marts/historical_flow.py, contexts/text_context.py가
  이미 재료 절반을 제공한다.)
