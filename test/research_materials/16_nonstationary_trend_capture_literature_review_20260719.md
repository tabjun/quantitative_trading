# 16번 방법론 근거 — 비정상 시계열의 "큰 변동까지 포함한 추세" 예측 + cross-sectional 딥러닝 문헌 리뷰

작성일: 2026-07-19 · 브랜치: `stock` · 목적: 16번 실험 설계 근거 및 참고문헌 섹션

> 조사 방법: repo `.mcp.json`의 arxiv MCP(`mcp__arxiv__*`)는 이번 세션에서 노출되지 않아
> **웹 검색 + arxiv.org 직접 확인**으로 대체했다. 각 논문은 arXiv abstract 페이지 또는
> 공식 학회 페이지에서 제목·저자·연도·ID를 대조해 실재를 확인했고, 확인 못 한 항목은
> "미확인"으로 명시했다. 지어낸 인용은 없다.
>
> 15번에서 확보된 냉정한 실측(`trend_corr` 최고 0.071, `r2` 전부 음수, `variance_ratio≈0.36`
> 진폭 압축, `large_move_da` 0.51~0.52)을 전제로, 16번은 "예측이 죽지도 폭주도 않는 상태"에서
> **큰 변동 폭까지 포함한 추세 신호를 실제로 끌어올릴** 방법론 근거를 세운다.

---

## 0. 이 리뷰가 답하는 질문과 16번 연결

연구 목적(CLAUDE.md 2.12): ① 학습 건전성(쉬운 해 붕괴 방지·과적합 관리·비정상성 가정 충족)
위에서 ② 예측 추세를 **작은 폭뿐 아니라 큰 변동까지** 잡고 ③ MDD를 생존 제약으로 함께 지킴.
데이터 축 기본값은 업비트 KRW 전 종목 15분봉.

이 리뷰는 그 세 축을 문헌으로 뒷받침한다.

- **비정상성 처리(1절)** → 학습 건전성 + "비정상성 가정 충족"
- **큰 변동/극단값 포착(2절)** → "큰 변동까지" + 진폭 압축(variance_ratio) 해소
- **cross-sectional 채널 처리(3절)** → "KRW 전 종목" 축 복귀의 설계 근거
- **평가 지표(4절)** → "정확도 단독이 아니라 정밀도·재현율"로 큰 변동 포착률 측정
- **과적합·학습곡선(5절)** → 과적합 관리, 11번 double descent 연결
- **암호화폐 특화(6절)** → 위 기법이 실제 crypto에서 쓰인 사례

---

## 1. 비정상성 처리 (Distribution Shift / Non-stationarity Normalization)

### 왜 필요한가 (공통 근거)

금융·crypto 15분봉은 평균·분산이 시간에 따라 바뀌는 **distribution shift**가 강하다. 학습
분포와 테스트 분포가 달라 모델이 과거 평균/스케일을 그대로 외우면(비정상성 미처리) 검증에서
무너진다. 아래 4기법은 "무엇을 정규화하고 무엇을 되돌리는가"가 서로 다르며, 이 차이가 곧
"진폭(큰 변동)을 살리느냐 죽이느냐"를 좌우한다 — 15번의 진폭 압축(variance_ratio≈0.36)과
직결되는 대목이다.

### 1.1 RevIN — Reversible Instance Normalization

- **출처**: Kim, Kim, Tae, Park, Choi, Choo, "Reversible Instance Normalization for Accurate
  Time-Series Forecasting against Distribution Shift", ICLR 2022.
  https://openreview.net/forum?id=cGDAkQo1C0p (확인)
- **무엇을 정규화하나**: 각 입력 **인스턴스(윈도우)** 단위로 평균·분산을 제거(normalize)해
  모델에 넣고, 출력에서 그 통계를 **다시 더해 복원(denormalize)**한다. 학습 가능한 affine
  파라미터(γ, β)를 둔다.
- **왜 비정상에 맞나**: 윈도우별 레벨/스케일 shift(비정상적 평균·분산)를 입력에서 잠깐
  제거하므로 모델은 "shift를 뺀 형태"만 학습하고, 복원 단계에서 각 윈도우 고유 스케일을 돌려준다.
- **핵심 아이디어 / 16번 반영 / 한 줄 근거**
  - 핵심: instance 단위 정규화-역정규화로 분포 이동 제거.
  - 16번: 15번의 `window_standard`는 RevIN의 경량판이었다. 16번에서 **정식 RevIN 레이어(γ,β
    학습)**를 backbone 앞뒤에 붙이고, 역정규화가 진폭을 얼마나 복원하는지 variance_ratio로 확인.
  - 근거: instance 통계 복원이 없으면 예측이 0 근처 평탄화로 붕괴(15번 관찰과 일치).

### 1.2 Non-stationary Transformer — Series Stationarization + De-stationary Attention

- **출처**: Liu, Wu, Wang, Long, "Non-stationary Transformers: Exploring the Stationarity in
  Time Series Forecasting", NeurIPS 2022. arXiv:2205.14415 (확인)
- **무엇을 정규화하나**: 두 모듈. (1) **Series Stationarization**은 입력 통계를 통일하고
  출력에서 복원(RevIN과 유사한 정규화-역정규화). (2) **De-stationary Attention**은 정규화로
  **사라진 비정상 정보**를 attention 점수에 다시 주입한다.
- **왜 비정상에 맞나**: 핵심 통찰 — 과도한 정상화는 "예측 가능성"은 올리지만, 정규화로
  깎여나간 비정상 정보(=버스티한 급변) 자체가 없어져 **큰 변동 예측을 오히려 죽인다**
  (over-stationarization 문제). De-stationary attention이 원본(비정규화) 시계열에서 배운
  attention을 근사해 그 급변 정보를 되살린다.
- **핵심 아이디어 / 16번 반영 / 한 줄 근거**
  - 핵심: 정규화하되, 정규화가 지운 급변 정보를 attention에 재주입.
  - 16번: 이것이 **15번 진폭 압축의 이론적 진단**이다. 단순 RevIN이 큰 변동을 못 살리면,
    De-stationary Attention 계열(또는 정규화 통계를 예측 head에 재입력)을 T계열 suite에 추가.
  - 근거: "정상화가 과하면 큰 변동을 못 잡는다"는 우리 15번 현상과 정확히 대응.

### 1.3 Dish-TS — Distribution Shift with Dual-CONET

- **출처**: Fan, Wang, Zhang, Zhou, Chen, "Dish-TS: A General Paradigm for Alleviating
  Distribution Shift in Time Series Forecasting", AAAI 2023. arXiv:2302.14829 (확인)
- **무엇을 정규화하나**: shift를 **intra-space(입력 윈도우 내부의 시간에 따른 이동)**와
  **inter-space(입력 윈도우 ↔ 예측 구간 사이의 이동)** 둘로 구분. **Dual-CONET**(BACK-CONET은
  lookback 통계, HORI-CONET은 horizon 통계)이 각 공간의 분포 계수를 **학습으로 추정**한다.
- **왜 비정상에 맞나**: RevIN은 입력 통계로 출력을 복원한다고 가정하지만, 비정상 시계열에서는
  **예측 구간의 통계가 입력 구간과 다르다**(inter-space shift). Dish-TS는 horizon 통계를 따로
  예측해 이 gap을 메운다.
- **핵심 아이디어 / 16번 반영 / 한 줄 근거**
  - 핵심: 입력-출력 통계가 다를 수 있다고 보고 horizon 통계를 별도 학습.
  - 16번: h-step 누적수익률(추세) target을 쓰는 우리 설정에서 h가 커질수록 입력↔horizon 통계
    괴리가 커진다. RevIN이 큰 h(16/64봉)에서 부족하면 Dish-TS식 horizon 통계 예측을 대안 축으로.
  - 근거: 15번이 h {1,4,16,64}를 이미 격자로 두므로 inter-space shift가 실제 변수다.

### 1.4 FAN — Frequency Adaptive Normalization

- **출처**: Ye, Deng, Zhang, Chen, "Frequency Adaptive Normalization For Non-stationary Time
  Series Forecasting", NeurIPS 2024. arXiv:2409.20371 (확인)
- **무엇을 정규화하나**: 평균·분산(1차 통계)만 제거하는 대신, **Fourier 변환으로 인스턴스별
  주요 주파수 성분**을 골라 제거하고, 입력↔출력 사이 그 주파수 성분의 변화를 별도 MLP로
  **예측**해 복원한다.
- **왜 비정상에 맞나**: RevIN류는 "레벨 shift(추세)"만 다루고 **계절성/주기적 비정상**은 못
  다룬다는 한계를 지적. FAN은 dynamic trend와 seasonal pattern을 함께 정규화한다. 4개 backbone,
  8개 벤치마크에서 MSE 7.76~37.90% 개선 보고.
- **핵심 아이디어 / 16번 반영 / 한 줄 근거**
  - 핵심: 통계가 아니라 지배적 주파수 성분을 정규화-예측-복원.
  - 16번: crypto 15분봉의 일중/주간 주기가 약하면 FAN 이득이 작을 수 있으나, model-agnostic이라
    최상위 backbone에 얹는 ablation 1셀로 저비용 검증 가능. FEDformer/frequency 계열과 함께 검토.
  - 근거: 우리 데이터는 "주기보다 shock이 강함"(기존 리뷰 4.5) → FAN은 확정 채택이 아니라
    조건부 후보로 둔다.

**1절 종합 (16번 반영 우선순위)**: RevIN(정식) → De-stationary(진폭 복원 실패 시) → Dish-TS(큰 h
실패 시) → FAN(주파수 조건부). 15번 `nonstationarity_screen`(T3) 축을 이 4계층으로 확장.

---

## 2. 큰 변동 / 극단값 포착 (Tail-aware & Distributional Forecasting)

### 왜 MSE/Huber 단독이 큰 변동을 과소예측하나 (핵심 근거)

- **MSE는 조건부 평균(E[y|x])을 추정**한다. 15분 수익률처럼 신호 대 잡음비가 0에 가까우면
  조건부 평균은 거의 0이다. 따라서 MSE 최소해는 "항상 0 근처"를 예측하는 것 — 이것이 15번의
  0 평탄화·진폭 압축(variance_ratio≈0.36)의 수학적 원인이다. 진폭을 키우면 잡음 구간에서
  제곱오차가 커지므로 모델은 **진폭을 줄이는 방향**으로 수렴한다(mean-reversion 착시).
- **Huber**는 큰 잔차에 선형 페널티를 줘 outlier에 덜 민감 → 극단값(큰 변동)의 gradient
  기여를 오히려 더 억누른다. 즉 outlier robustness가 "큰 변동을 무시해도 된다"로 작동해
  진폭 압축을 강화한다.
- 결론: **평균을 맞히는 손실로는 큰 변동을 살릴 수 없다.** 분위수/분포/방향/tail-가중 손실이
  필요하다.

### 2.1 Quantile Regression / Pinball Loss

- **출처(고전)**: Koenker & Bassett, "Regression Quantiles", Econometrica 1978 (분위수 회귀 원전).
  신경망 적용·pinball loss 설명은 널리 정착된 표준(예: Wikipedia "Quantile regression",
  https://en.wikipedia.org/wiki/Quantile_regression 확인). 특정 arXiv 단일 출처보다 표준 기법.
- **핵심 아이디어**: MSE 대신 tilted absolute (pinball) loss를 쓰면 조건부 **평균이 아니라
  조건부 분위수**(τ=0.05, 0.5, 0.95 등)를 학습. 여러 τ를 동시에 예측하면 예측 구간(분포 폭)을
  직접 얻는다.
- **16번 반영**: point head를 다중 분위수 head(예: {0.1,0.25,0.5,0.75,0.9})로 교체. **0.9-0.1
  분위 폭이 곧 예측된 변동 폭** → 큰 변동을 명시적으로 모델링. 중앙값(0.5)은 방향, 바깥 분위는 진폭.
- **한 줄 근거**: 평균이 0으로 붕괴해도 상·하위 분위는 큰 변동을 표현할 자유도를 가진다.

### 2.2 Distributional Forecasting — DeepAR / Gaussian·Student-t Head

- **출처**: Salinas, Flunkert, Gasthaus, "DeepAR: Probabilistic Forecasting with Autoregressive
  Recurrent Networks", 2017 → International Journal of Forecasting 2020. arXiv:1704.04110 (확인)
- **핵심 아이디어**: 점 하나가 아니라 **분포의 파라미터**(가우시안이면 μ, σ; Student-t면 ν까지)를
  네트워크가 출력하고, **음의 로그우도(NLL)**로 학습. σ가 커질 수 있으므로 변동을 명시적으로
  모델링. Student-t head는 **두꺼운 꼬리(fat tail)**를 표현 → crypto의 극단 급변에 적합.
- **16번 반영**: 회귀 head를 **Student-t NLL head**로 교체하는 축을 objective_screen(T2)에 추가.
  σ, ν가 큰 변동 구간에서 실제로 커지는지 진단. RevIN과 결합 시 μ는 shift-free, σ는 진폭 담당.
- **한 줄 근거**: fat-tail 분포 head는 정규분포 가정보다 crypto 극단값을 덜 과소예측한다.

### 2.3 Tail-Weighted / Imbalanced-Regression Loss

- **출처(대표)**:
  - Rudy & Sapsis, "Output-weighted and relative entropy loss functions for deep learning
    precursors of extreme events", 2021. arXiv:2112.00825 (확인)
  - "Boosting Time Series Prediction of Extreme Events by Reweighting and Fine-tuning",
    2024. arXiv:2409.14232 (확인)
  - imbalanced regression 일반론: Ribeiro & Moniz, "Imbalanced regression and extreme value
    prediction", Machine Learning (Springer) 2020 (확인)
- **핵심 아이디어**: 극단값은 데이터에서 희소 → 표준 손실이 무시. **|y|가 큰 샘플에 더 큰
  가중치**(inverse-frequency 또는 output-weighting)를 줘 큰 변동의 gradient 기여를 키운다.
- **16번 반영**: 15번 T7(amplitude) 축의 정식 근거. Huber/MSE에 **tail weight w(y)=1+α·|y|/σ**
  를 곱한 weighted loss를 objective 후보로. large_move_da·variance_ratio가 오르는지 검증.
- **한 줄 근거**: 큰 변동 구간에 손실 가중을 실어야 모델이 그 구간을 "포기하지 않는다".

### 2.4 (참고) 방향·거래 지향 손실 — GMADL/MADL

- **출처**: Michańków, Sakowski, Ślepaczuk, "Generalized Mean Absolute Directional Loss as a
  Solution to Overfitting and High Transaction Costs in ML Models Used in High-Frequency
  Algorithmic Investment Strategies", 2024. arXiv:2412.18405 (확인, 저자·연도 대조 완료).
  선행 MADL도 동일 그룹.
- **핵심 아이디어**: 수익률의 **방향(부호)**과 **크기**를 함께 보상하되, 거래 수를 줄이도록
  parameterize → 과적합·거래비용 동시 완화. Transformer/LSTM/RNN에서 MSE류보다 우수 보고.
- **16번 반영**: objective_screen(T2)의 방향 보조항 근거. 단 GMADL은 "예측 정확도"보다
  "거래 성과"에 최적화된 손실이므로, 예측축(trend_corr) 개선보다는 **T5 gate_fusion 정책 평가**
  단계에서 비교하는 것이 정합적이다.
- **한 줄 근거**: 방향+진폭 보상 손실이 0-예측 붕괴를 직접 페널티한다.

---

## 3. 채널 / 종목 축 처리 (Cross-sectional, KRW 전 종목)

### 왜 중요한가 (연구 목적 직결)

데이터 축 기본값이 "KRW 전 종목 15분봉"이므로, 여러 종목을 **어떻게 한 모델에 담느냐**가
16번의 핵심 설계 결정이다. 두 축: (a) 종목을 독립 채널로 볼지 함께 볼지(CI vs CD),
(b) 절대 수익률을 예측할지 종목 간 상대순위를 예측할지(pointwise vs cross-sectional ranking).

### 3.1 Channel-Independent (CI) — PatchTST

- **출처**: Nie, Nguyen, Sinthong, Kalagnanam, "A Time Series is Worth 64 Words: Long-term
  Forecasting with Transformers", ICLR 2023. arXiv:2211.14730 (확인)
- **핵심 아이디어**: 각 변수/종목을 **독립 univariate 시계열**로 보고 **공유 encoder**에
  통과(channel-independence). patching으로 긴 lookback을 저비용 처리.
- **장점**: 종목 간 노이즈 상호작용을 배제 → **과적합 위험↓, robustness↑**. 종목이 많을 때
  파라미터 공유로 데이터 효율↑(전 종목이 한 encoder를 함께 학습 = 사실상 데이터 증강).
- **단점**: 종목 간 상관(BTC↔알트 동조화, 섹터 로테이션)을 **구조적으로 못 본다**.
- **16번 반영**: KRW 전 종목을 CI로 한 encoder에 태우는 것이 **1차 기본 설정**. 종목 ID
  임베딩을 더해 종목 특성을 소량 주입 가능.

### 3.2 Channel-Dependent (CD) — iTransformer (변수 토큰)

- **출처**: Liu, Hu, Zhang, Wu, Wang, Ma, Long, "iTransformer: Inverted Transformers Are
  Effective for Time Series Forecasting", ICLR 2024. arXiv:2310.06625 (확인)
- **핵심 아이디어**: timestamp가 아니라 **각 변수(종목)의 전체 시계열을 하나의 토큰**으로 삼아
  **변수 간 attention**을 적용 → 종목 간 상관을 attention이 직접 학습. 비동기(time-misalignment)
  거동도 흡수.
- **장점**: cross-sectional 동조화/전이(예: BTC 급락 → 알트 동반)를 모델링 → 표현력↑.
- **단점**: 변수(종목)가 많고 노이즈가 크면 spurious correlation 과적합 위험. KRW 전 종목처럼
  수백 개면 토큰 수 폭증.
- **16번 반영**: CI 기본선을 **CD(iTransformer)와 A/B**. cross-sectional 상관이 실제 예측
  이득을 주는지가 연구 질문. 종목 과다 시 시총 상위 N개로 제한하거나 종목 군집 후 그룹 토큰화.

### 3.3 Cross-Sectional Ranking / Relative Strength

- **출처**: Feng, He, Wang, Luo, Liu, Chua, "Temporal Relational Ranking for Stock Prediction"
  (RSR), ACM TOIS 2019. arXiv:1809.09441 (확인)
- **핵심 아이디어**: 절대 수익률 회귀 대신, 매 시점 **종목들을 수익률 기준으로 순위화**하는
  것을 학습(pointwise + pairwise ranking loss). Temporal Graph Convolution으로 종목 관계를
  시간에 따라 모델링. "누가 가장 오를까"만 맞히면 되므로 절대 스케일 예측 부담이 없다.
- **왜 우리 문제에 맞나**: 15분 절대 수익률은 SN비≈0이라 회귀가 붕괴하지만, **종목 간 상대
  강도(relative strength)**는 절대 레벨보다 신호가 살아있는 경우가 많다(cross-sectional
  momentum). 큰 변동 종목이 순위 상·하단에 몰리므로 "큰 변동 포착"과도 부합.
- **16번 반영**: 절대 수익률 회귀(15번 방식)와 **cross-sectional ranking objective**를 병렬
  축으로. 평가는 순위 상관(Spearman/IC), 상·하위 분위 포트폴리오 스프레드. **KRW 전 종목 축을
  살리는 가장 자연스러운 objective**다.
- **한 줄 근거**: 절대 예측이 붕괴해도 상대 순위는 학습 가능한 신호를 남긴다(IC 문헌 정착).

**3절 종합**: CI(PatchTST, 기본선) → CD(iTransformer, 상관 이득 검증) → Ranking(RSR, 상대강도
축). 세 가지를 채널 처리 suite로 두면 "KRW 전 종목"을 진단 목적 단일 축으로 좁히지 않고
정식 cross-sectional 실험으로 승격할 수 있다.

---

## 4. 평가 지표 (큰 변동 포착을 정밀도·재현율/방향성/분포로)

연구 목적: "작은 폭만 맞추는 것은 정확한 것이 아니다 — 정확도 단독이 아니라 정밀도·재현율."
아래 지표군은 15번 지표(mae_zero_ratio, mase_momentum, trend_corr, large_move_da,
variance_ratio)를 확장·정당화한다.

### 4.1 큰 변동 구간 = 분류 문제로 (정밀도/재현율/F1)

- **핵심 아이디어**: "|실제 수익률| 상위 25%인 큰 변동"을 **양성 클래스**로 정의하고, 모델이
  "큰 변동이라고 예측한 것 중 실제 큰 변동 비율(정밀도)"과 "실제 큰 변동 중 잡아낸 비율
  (재현율)"을 측정. 방향까지 맞았는지는 large_move_da로.
- **16번 반영**: 15번 `large_move_da`를 **precision/recall/F1로 분해**. 진폭 압축 모델은
  재현율이 구조적으로 낮게 나옴(큰 변동을 아예 예측 안 함) → 진폭 압축을 정량 노출.
- **한 줄 근거**: 정확도(전체 DA)는 잦은 소폭 구간에 지배되어 큰 변동 실패를 가린다.

### 4.2 MASE (Mean Absolute Scaled Error)

- **출처**: Hyndman & Koehler, "Another look at measures of forecast accuracy",
  International Journal of Forecasting 22(4):679-688, 2006. https://robjhyndman.com/papers/mase.pdf (확인)
- **핵심 아이디어**: 오차를 **naive(직전값 지속) 예측의 in-sample MAE로 나눠** 스케일 독립화.
  <1이면 naive보다 우수. 여러 종목/스케일을 함께 비교할 때 표준(→ KRW 전 종목에 필수).
- **16번 반영**: 15번 `mase_momentum`의 정식 근거. 전 종목 패널에서 종목별 스케일이 다르므로
  MASE로 통일해 랭킹.
- **한 줄 근거**: MAPE·RMSE는 스케일·0근처에서 붕괴, MASE는 패널 비교에 안전.

### 4.3 분포 예측 평가 — CRPS / Pinball

- **출처**: Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation",
  JASA 2007 (CRPS의 proper scoring rule 근거, 확인). CRPS는 MAE의 확률예측 일반화.
- **핵심 아이디어**: 2절의 분위/분포 head를 쓰면 점오차가 아니라 **예측 분포 전체의 품질**을
  평가해야 한다. **CRPS**는 예측 CDF와 실제값의 거리(proper scoring rule → 정직한 분포를
  낼 때 최소). **pinball loss 합**은 다중 분위 예측의 평가/학습 지표.
- **16번 반영**: quantile/DeepAR head를 도입하면 CRPS·pinball을 주 지표로. 큰 변동 구간에서만
  계산한 **tail-CRPS**로 극단 포착을 별도 평가.
- **한 줄 근거**: 분포 예측을 MAE로만 보면 진폭·불확실성 정보를 버린다.

### 4.4 Variance Ratio (예측분산 / 실제분산)

- **핵심 아이디어**: `Var(예측)/Var(실제)`. ≈1이면 진폭 건강, ≪0.1이면 평탄화(진폭 압축),
  ≫10이면 폭주. 15번에서 이미 건강 gate(0.05~20)로 사용 중. (※ 계량경제의 Lo-MacKinlay
  variance ratio test와는 다른, 우리 진단용 진폭 비율 — 보고서에 정의 명시 필요.)
- **16번 반영**: 랭킹의 건강 gate로 유지 + 목표를 "0.1~10 통과"에서 **"1에 근접"**으로 상향
  (진폭을 죽이지 않는 것을 성공 조건으로). 2절 손실들이 실제로 variance_ratio를 1쪽으로
  미는지가 T7의 성공 기준.
- **한 줄 근거**: 평탄한 예측 위의 좋은 MAE/MDD는 성과가 아니다(CLAUDE.md 2.12 직접 대응).

**4절 종합**: 랭킹 지표를 {건강 gate: variance_ratio≈1} → {큰 변동: tail precision/recall/F1,
tail-CRPS} → {상대: MASE, cross-sectional IC} 3층으로. "정밀도·재현율" 요구를 정식화.

---

## 5. 과적합 · 학습 곡선 진단 (11번 double descent 연결)

### 5.1 Early Stopping / 학습·검증 곡선 해석

- **핵심 아이디어(표준)**: train loss는 내려가는데 validation loss가 올라가면 과적합 시작 →
  최소 validation 지점에서 early stop. 우리 문제 특유의 함정: **train도 val도 안 내려가면서
  예측이 0으로 수렴**하는 경우가 있는데, 이건 과적합이 아니라 **쉬운 해로의 붕괴**다(15번).
  따라서 loss 곡선만으로 판단하지 말고 variance_ratio·trend_corr 곡선을 **함께** 그린다.
- **16번 반영**: 매 epoch마다 loss뿐 아니라 variance_ratio·large_move_recall을 로깅해
  "붕괴 vs 과적합 vs 학습중"을 구분. early stopping 기준을 val_loss 단독이 아니라
  **val_loss + variance_ratio 건강**의 복합 기준으로.
- **한 줄 근거**: 0-붕괴는 val_loss가 낮아 early stopping이 오히려 붕괴 해를 채택할 수 있다.

### 5.2 Deep Double Descent (11번과의 관계)

- **출처**: Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever, "Deep Double Descent: Where
  Bigger Models and More Data Hurt", ICLR 2020. arXiv:1912.02292 (확인)
- **핵심 아이디어**: 모델 크기(또는 **epoch 수**)를 키우면 test error가 한 번 나빠졌다가
  (interpolation threshold 부근) 다시 좋아진다. 고전적 bias-variance U자와 다른 현상 →
  "과적합처럼 보이는 구간"이 실제로는 double descent의 중간일 수 있다. 데이터를 늘려도
  특정 구간에선 나빠질 수 있음.
- **16번 반영**: 11번에서 다룬 capacity 진단을 계승. 모델 폭/깊이/epoch를 격자로 두되,
  "Shallow but Wide"(CLAUDE.md 4.3: 1~2층, width 64~128) 원칙 안에서 **epoch-wise double
  descent**를 관측해 early stopping 지점이 첫 descent인지 확인. 15번 T6의 데이터 규모 효과
  (`--max-windows/--stride`)도 double descent 관점에서 재해석.
- **한 줄 근거**: 검증 곡선의 일시적 악화를 무조건 과적합으로 오판하면 최적 용량을 놓친다.

---

## 6. 암호화폐 특화 — 위 기법의 실제 사용례

### 6.1 고빈도 Bitcoin + 손실함수 (우리 문제와 최근접)

- **출처**: Stefaniuk & Ślepaczuk, "Informer In Algorithmic Investment Strategies on High
  Frequency Bitcoin Data", 2025. arXiv:2503.18096 (확인)
- **무엇을 했나**: 5/15/30분 BTC에 Informer를 **RMSE / Quantile / GMADL** 세 손실로 학습,
  예측을 매매로 바꿔 Buy&Hold·MACD·RSI와 비교. **RMSE는 5/15분에서 거래비용보다 작은 0 근처
  예측**을 내는 문제를 명시(= 우리 15번 0-붕괴와 동일 현상). GMADL·고빈도에서 성과 우위 보고.
- **16번 반영**: (a) RMSE 0-붕괴는 crypto 고빈도의 알려진 실패 모드 → 우리만의 문제가 아님을
  보고서에 근거로. (b) Quantile·GMADL 손실이 그 해법으로 실증됨 → 2절·2.4절 채택 정당화.
- **한 줄 근거**: 동일 데이터 성격에서 손실함수 교체가 0-붕괴를 실제로 해결한 선례.

### 6.2 crypto 딥러닝 예측 리뷰 (모델·평가 프로토콜)

- **출처**: Wu et al., "Review of deep learning models for crypto price prediction:
  implementation and evaluation", 2024. arXiv:2405.11431 (확인)
- **무엇을 했나**: LSTM/CNN/Transformer 계열을 univariate/multivariate multi-step close 예측으로
  비교. pre-COVID/COVID **regime별** 시나리오와 **volatility analysis**를 별도 절로. Adam 학습
  세부까지 명시.
- **16번 반영**: regime split(급등/급락/횡보)별로 큰 변동 포착을 나눠 평가하는 절을 16번
  보고서에 포함할 근거. crypto 논문은 volatility를 1급 평가 대상으로 둔다는 관행 확인.
- **한 줄 근거**: crypto 예측 논문은 "가격 정확도"가 아니라 regime·변동성으로 성과를 본다.

### 6.3 다종목(cross-sectional) crypto 딥러닝

- **출처**: "From LSTM to GPT-2: Recurrent and Transformer-Based Deep Learning Architectures
  for Multivariate High-Liquidity Cryptocurrency Price Forecasting", MDPI Symmetry 18(1):32,
  2026 (확인). BTC/ETH/XRP/XLM/SOL 5종목을 LSTM·GPT-2·Informer·Autoformer·TFT·Vanilla
  Transformer + 계량경제 4모델로 대칭 비교.
- **참고(미확인 세부)**: whale/온체인 기반 BTC 변동성 스파이크 예측(Synthesizer Transformer,
  arXiv:2211.08281 — 제목·ID는 검색에 나타났으나 본문 정밀 확인은 미수행, "미확인" 표기).
- **16번 반영**: 다종목을 함께 다루는 crypto 딥러닝이 이미 표준 방향임을 보여줌 → KRW 전 종목
  축이 특이하지 않음을 뒷받침. 단 대부분 소수 고유동성 종목에 한정 → **KRW 전 종목(수백 개)
  cross-sectional은 상대적 공백**이라 우리 기여점이 될 수 있다.
- **한 줄 근거**: 다종목 crypto 딥러닝은 존재하나 "전 종목 cross-sectional ranking"은 희소.

---

## 7. 16번 실험 설계 반영 요약 (액션)

| 축 | 15번 상태 | 16번 확장 (문헌 근거) |
|---|---|---|
| 비정상성 정규화 | window_standard(RevIN 경량) | 정식 RevIN → De-stationary attention → Dish-TS(큰 h) → FAN(주파수 조건부) |
| 손실/head | Huber + 보조항, 진폭 압축(vr≈0.36) | 다중 분위(pinball) / Student-t NLL(DeepAR) / tail-weighted / (정책단 GMADL) |
| 채널·종목 | BTC 단일 + T8 다자산 확장 | CI(PatchTST 기본) vs CD(iTransformer) vs cross-sectional ranking(RSR) — KRW 전 종목 정식화 |
| 평가 | large_move_da, variance_ratio | tail precision/recall/F1, tail-CRPS, MASE, cross-sectional IC, variance_ratio 목표 ≈1 |
| 과적합 진단 | seed·규모 회차 | val_loss+variance_ratio 복합 early stop, epoch-wise double descent(11번 계승) |

**핵심 서사**: 15번은 "예측이 죽지도 폭주도 않는 상태"까지 왔으나 진폭이 압축(vr≈0.36)돼 큰
변동을 못 잡았다. 16번은 (1) 정규화가 지운 급변 정보를 되살리고(De-stationary/Dish-TS/FAN),
(2) 평균 손실을 분위·분포·tail 손실로 바꿔 큰 변동을 명시 모델링하고, (3) KRW 전 종목을
cross-sectional ranking으로 정식화해, "큰 변동 폭까지 포함한 추세"를 정밀도·재현율로 측정한다.

---

## 8. 참고문헌 (arXiv/venue 확인 완료 = ✅, 미확인 = ⚠)

### 비정상성 처리
- ✅ RevIN — Kim et al., ICLR 2022. https://openreview.net/forum?id=cGDAkQo1C0p
- ✅ Non-stationary Transformers — Liu et al., NeurIPS 2022. arXiv:2205.14415
- ✅ Dish-TS — Fan et al., AAAI 2023. arXiv:2302.14829
- ✅ FAN — Ye et al., NeurIPS 2024. arXiv:2409.20371

### 큰 변동 / 극단값
- ✅ Quantile Regression — Koenker & Bassett, Econometrica 1978 (pinball loss 표준)
- ✅ DeepAR — Salinas et al., IJF 2020 / arXiv:1704.04110
- ✅ Output-weighted loss for extremes — Rudy & Sapsis, arXiv:2112.00825
- ✅ Reweighting extremes — arXiv:2409.14232 (2024)
- ✅ Imbalanced regression & extreme value — Ribeiro & Moniz, Machine Learning (Springer) 2020
- ✅ GMADL — Michańków, Sakowski, Ślepaczuk, arXiv:2412.18405 (2024)

### 채널 / cross-sectional
- ✅ PatchTST (CI) — Nie et al., ICLR 2023. arXiv:2211.14730
- ✅ iTransformer (CD, 변수 토큰) — Liu et al., ICLR 2024. arXiv:2310.06625
- ✅ RSR (temporal relational ranking) — Feng et al., ACM TOIS 2019. arXiv:1809.09441

### 평가 지표
- ✅ MASE — Hyndman & Koehler, IJF 2006. https://robjhyndman.com/papers/mase.pdf
- ✅ CRPS / proper scoring — Gneiting & Raftery, JASA 2007

### 과적합 / double descent
- ✅ Deep Double Descent — Nakkiran et al., ICLR 2020. arXiv:1912.02292

### 암호화폐 특화
- ✅ Informer HF Bitcoin (RMSE/Quantile/GMADL) — Stefaniuk & Ślepaczuk, arXiv:2503.18096 (2025)
- ✅ Crypto DL review — Wu et al., arXiv:2405.11431 (2024)
- ✅ Multivariate high-liquidity crypto (5 coins) — MDPI Symmetry 18(1):32, 2026
- ⚠ Bitcoin volatility spikes from whale/on-chain (Synthesizer Transformer) — arXiv:2211.08281
  (제목·ID 검색 확인, 본문 정밀 대조 미수행 → 인용 전 재확인 필요)
