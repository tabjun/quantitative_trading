# 모델 카탈로그 — 아키텍처 특징과 문헌 근거

이 문서는 모델의 **특징·문헌 근거와 현재 채택 상태를 한 행에 함께** 적는다. 계열별로
표 하나씩 묶고, **처리 방식** 칸이 입력을 읽는 법(순차·재귀/병렬 어텐션/사전학습 등)을 나타내며, **비고** 칸이 상태(기존 유지/공식 구현 교체/신규 도입/미채택)와 적용
연구 번호를 담당한다 — 모델이 바뀔 때마다 새 문서를 만들지 않고 그 행만 갱신한다.
실제로 그 연구가 무엇을 했는지(실행 결과)는 `test/results/{N}_*/`에 남는다. 실험 번호와
무관하게 누적되는 참조용 카탈로그라 이 문서 자체엔 번호를 붙이지 않는다
(`test/research_materials/`의 비번호 문서 관례 — `forecasting_methodology_literature_
review_20260613.md` 등과 같은 층위).

새 모델을 조사할 때마다 해당 계열 표에 행을 추가한다. 기존 행은 내용이 틀렸거나
상태(비고)가 바뀌었을 때만 고친다.

---

## 1. 통계·계량 모델

| 모델 | 문헌 | 특징 | 구현 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HAR-RV | Corsi(2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, *Journal of Financial Econometrics* 7(2) | 로그 실현변동성을 과거 1봉·1일(평균)·1주(평균) 세 항의 선형회귀로 설명. 변동성의 장기기억을 복잡한 분수차분 없이 "이질적 투자기간을 가진 거래자가 서로 다른 시간축을 본다"는 경제적 해석으로 근사. 파라미터 3개뿐이라 과적합 위험이 거의 없어 "이기기 어려운 표준 벤치마크"로 쓰임 | `sklearn.linear_model.LinearRegression`(자체 특성 생성) — 모델이 단순 선형식이라 자체 구현이 곧 공식 사양 | 지연 특성 회귀(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| GARCH-t | Bollerslev(1986), *Generalized Autoregressive Conditional Heteroskedasticity*, *Journal of Econometrics* 31(3); Bollerslev(1987)이 t분포로 확장 | 조건부분산이 과거 조건부분산과 과거 제곱잔차의 선형결합으로 재귀적으로 갱신. 조건부 분포를 정규분포 대신 t분포로 둬 금융수익률의 두꺼운 꼬리(fat tail)를 반영 | `arch` 패키지(공식 유지보수 중인 GARCH 추정 라이브러리) | 순차·재귀(통계) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| MS-GARCH | Hamilton(1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*, *Econometrica* 57(2) + Gray(1996)·Klaassen(2002) collapsing 근사 | 국면(regime)이 관측되지 않는 잠재변수. Hamilton 필터로 매 시점 국면확률을 추정하고, Gray의 축약 절차로 국면경로가 2^t가지로 폭발하는 문제를 피함. 두 국면 모두 조건부밀도는 t분포(GARCH-t와 비교 가능) | `engine/regime_garch.py`(자체 구현 — 경로 축약 근사 자체가 논문의 수식이라 패키지 없이 직접 구현하는 것이 표준) | 순차·재귀(통계) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| TAR-GARCH | 임계값 자기회귀(SETAR, Threshold Autoregression) 전통을 GARCH 조건부분산에 적용한 레짐전환 GARCH 계열 | MS-GARCH와 달리 국면이 관측 가능한 임계변수(과거 수익률 부호·크기 등)가 문턱값을 넘는지로 즉시·결정론적으로 전환. 필터링이 필요 없어 MS-GARCH보다 계산이 단순하고 해석이 직관적 | `engine/regime_garch.py`(자체 구현) | 순차·재귀(통계) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |

---

## 2. 선형·커널·트리 기반 지도학습

| 모델 | 특징 | 구현 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| Linear / Ridge | 표준 (정규화) 선형회귀 | `sklearn.linear_model` | 특성 기반 회귀(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| KernelRidge-RBF | RBF 커널로 비선형 관계를 커널 릿지로 학습 | `sklearn.kernel_ridge.KernelRidge` | 특성 기반 회귀(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| SVR-RBF | RBF 커널 서포트벡터회귀 | `sklearn.svm.SVR`(libsvm) | 특성 기반 회귀(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| Nystroem+Ridge | Nystroem 커널근사로 대표본에서 커널릿지를 선형 비용으로 근사 | `sklearn.kernel_approximation.Nystroem` + `Ridge` | 특성 기반 회귀(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| LightGBM | 히스토그램 기반 그래디언트 부스팅(leaf-wise 성장) | 공식 `lightgbm` 패키지(Microsoft) | 특성 기반 트리(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| XGBoost | 그래디언트 부스팅(level-wise 성장 + 정교한 정규화) | 공식 `xgboost` 패키지 | 특성 기반 트리(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| HistGBM | 히스토그램 기반 그래디언트 부스팅(sklearn 내장) | `sklearn.ensemble.HistGradientBoostingRegressor` | 특성 기반 트리(비신경) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| GARCH+LightGBM | GARCH 조건부분산을 트리 특성으로 투입하는 하이브리드 | 위 두 구현의 조합(자체 결합 로직) | 순차·재귀(통계) | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |

이 계열은 아키텍처 자체가 잘 정립된 표준 알고리즘이라 원 논문 재현 이슈가 없다 — 전부
유지보수 중인 공식 패키지를 그대로 쓴다.

---

## 3. 순환신경망(RNN) 계열

| 모델 | 문헌 | 특징 | 구현 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GRU | Cho 외(2014), *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, EMNLP | LSTM보다 게이트 수가 적은(리셋·업데이트 게이트 2개) 경량 순환 구조. 장기 의존성을 학습하면서도 파라미터가 LSTM보다 적음 | `torch.nn.GRU`(PyTorch 공식 내장) 위에 자체 입출력 헤드 | 순차·재귀 | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |
| LSTM | Hochreiter & Schmidhuber(1997), *Long Short-Term Memory*, *Neural Computation* 9(8) | 입력·망각·출력 3게이트와 별도 cell state로 장기 의존성의 그래디언트 소실 문제를 완화. 시계열·언어모델링의 고전적 표준 | `torch.nn.LSTM`(PyTorch 공식 내장) | 순차·재귀 | 기존 유지 · 적용 연구 24, 25, 26(시간 기준 재평가) |

---

## 4. Transformer·경량 혼합(Mixer) 계열 — neuralforecast 공식 구현

아래 전부 Nixtla의 `neuralforecast` 패키지(유지보수 중인 공식 오픈소스, pip 설치)가 원
논문 아키텍처를 구현해 제공한다. 자체 재구현(`engine/models.py`의 `~Like` 클래스들)은
AGENTS.md 2.13(모델명 Like 금지 — 검증된 구현 사용 의무)의 계기가 된 PatchTSTLike 채널혼합
오류 이후, 공식 구현이 있으면 그쪽을 쓰는 것으로 방향을 바꿨다.

| 모델 | 문헌 | 특징 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| PatchTST | Nie 외(2023), *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers*, ICLR 2023 | 시계열을 겹치는 패치로 나눠 토큰화(어텐션 연산량 절감), **채널독립(channel-independence)** — 다변량의 각 채널을 완전히 별도로 처리해 채널 간 거짓 상관을 학습하지 않게 함. 이 채널독립이 원 논문의 핵심 설계 | 병렬 어텐션(패치 임베딩) | 폐기된 자체구현(PatchTSTLike, 채널혼합 오류)을 공식 구현으로 교체 · 적용 연구 24(자체구현판, 제외됨) → 27(공식판, 예정) |
| iTransformer | Liu 외(2024), *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024 | 어텐션의 토큰 축을 "시간"이 아니라 "변수(채널)"로 뒤집음 — 각 변수의 전체 시계열을 하나의 토큰으로 취급해, 어텐션이 변수 간 관계를, 피드포워드가 시간 축을 처리 | 병렬 어텐션(변수 토큰) | 폐기된 자체구현(ITransformerLike)을 공식 구현으로 교체 · 적용 연구 24(자체구현판, 제외됨) → 27(공식판, 예정) |
| TCN | Bai, Kolter & Koltun(2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv:1803.01271 | 인과적(causal) 팽창 합성곱(dilated convolution)을 쌓아 수용영역을 지수적으로 확대. RNN의 순차 처리 병목 없이 병렬화 가능 | 병렬 합성곱 | 신규 도입(`engine/models.py`의 자체구현 TCNForecaster가 미사용으로 남아 있던 것을 공식 구현으로 대체) · 적용 연구 27(예정) |
| DLinear / NLinear | Zeng 외(2023), *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023 | "복잡한 Transformer가 단순 선형모델을 못 이긴다"는 문제의식의 대조군. DLinear는 추세·계절 성분 분해 후 각각 선형층으로 예측, NLinear는 입력 마지막 값을 빼는 정규화 후 선형층으로 예측(분포 shift에 강건) | 병렬 선형 | 제외(2026-10-05 결정: 선형 예측기라 2026-09-07 선형성 기각 결정에 따라 Linear·Ridge·HAR-RV와 함께 뺌) · 적용 연구 없음 |
| Autoformer | Wu 외(2021), *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*, NeurIPS 2021 | 어텐션 대신 자기상관(auto-correlation) 메커니즘으로 주기적 의존성을 직접 탐색, 추세-계절 분해 블록을 반복적으로 넣어 분해와 예측을 함께 학습 | 병렬 자기상관 어텐션 | 신규 도입(`engine/models.py`의 자체구현 AutoformerLike가 미사용으로 남아 있던 것을 공식 구현으로 대체) · 적용 연구 27(예정) |
| TimesNet | Wu 외(2023), *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*, ICLR 2023 | 1차원 시계열을 FFT로 찾은 주요 주기에 맞춰 2차원(주기×주기 내 위치) 텐서로 접어 2D 합성곱으로 시간 내·시간 간 변동을 함께 포착 | 병렬 2D 합성곱 | 신규 도입(`engine/models.py`의 자체구현 TimesNetLike가 미사용으로 남아 있던 것을 공식 구현으로 대체) · 적용 연구 27(예정) |
| TimeXer | Wang 외(2024), *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables*, NeurIPS 2024 | 내생(예측 대상) 변수와 외생 변수를 별도 임베딩 경로로 분리 처리 후 교차 어텐션으로 결합 — 외생변수가 있는 예측 문제에 특화 | 병렬 어텐션(외생 교차) | 신규 도입(`engine/models.py`의 자체구현 TimeXerLike가 미사용으로 남아 있던 것을 공식 구현으로 대체) · 적용 연구 27(예정) |
| VanillaTransformer | Vaswani 외(2017), *Attention Is All You Need*, NeurIPS 2017 | 패치·분해·역전치 등 시계열 특화 설계 없이 표준 어텐션 인코더-디코더를 그대로 사용 | 병렬 어텐션 | 공식 구현 있으나 미채택 — GRU·LSTM·PatchTST가 이미 순환·어텐션 계열을 대표해 중복 비교 가치가 낮다고 판단 |

---

## 5. 합성곱·선형어텐션 계열 — 공식 GitHub 구현(neuralforecast 미제공)

| 모델 | 문헌 | 특징 | 공식 구현 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ModernTCN | Luo & Wang(2024), *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*, ICLR 2024 Spotlight | 어텐션 없이 순수 합성곱만으로 Transformer급 성능을 노린 구조. depthwise separable 합성곱으로 시간·변수·특징 세 축을 독립적으로 확장(ConvNeXt의 시계열판에 가까움) | `github.com/luodhhh/ModernTCN`(저자 공식, pip 패키지 없음) | 병렬 합성곱 | 신규 도입(`engine/models.py`의 자체구현 ModernTCNLike를 공식 구현으로 대체, 어댑터 작성 필요) · 적용 연구 27(예정) |
| Mamba / S-Mamba | 원형: Gu & Dao(2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752(범용 시퀀스 모델, 시계열·비정상성 특화 구조 아님). 시계열 적용: Wang 외(2024), *Is Mamba Effective for Time Series Forecasting?*, *Neurocomputing*(2024) | 양방향(bidirectional) Mamba 블록으로 변수 간 상관을, 피드포워드로 시간 의존성을 각각 처리. 긴 lookback을 선형 비용(O(n))으로 봐서 장기 의존성(변동성 지속성 등)을 포착하는 데 강점 — 단, MS-GARCH·TAR-GARCH처럼 국면전환을 명시적으로 모델링하는 구조는 아님 | `github.com/wzhwzhwzh0921/S-D-Mamba`(저자 공식, pip 패키지 없음) | 선택적 상태공간(학습 병렬·추론 재귀) | 신규 도입(자체구현 MambaLike를 공식 구현으로 대체) · 적용 연구 27 · **설치**: 공식 요구사항이 `torch==2.0.1`·`mamba-ssm==1.2.0`이고 서버에 `nvcc`가 없어 컴파일 설치는 실패했으나(`pip` 빌드 오류), 저자 버전에 맞춰 미리 컴파일된 wheel(`mamba_ssm-1.2.0+cu118torch2.0`, `causal_conv1d-1.2.0.post2`)로 격리 venv `.venvs/smamba_py310_20261006`(Python 3.10, torch 2.0.1+cu118, transformers 4.40.1 고정)을 구성했다(`sudo`·시스템 변경 없음, 메인 venv 무관). 저장소는 `third_party/S-D-Mamba`(커밋 e7e8bf0), 러너는 `test/models/27_smamba_run.py` |

---

## 6. 시계열 파운데이션 모델 (사전학습 + zero-shot)

언어모델처럼 대규모·다종 시계열로 사전학습한 뒤, 우리 데이터에 **재학습 없이(zero-shot)**
바로 추론한다는 점이 1~5절의 모든 모델과 다르다.

| 모델 | 문헌 | 특징 | 공식 구현 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Chronos-Bolt | Ansari 외(2024), *Chronos: Learning the Language of Time Series*, arXiv:2403.07815(Bolt는 후속 고속 증류 버전) | T5 기반 인코더-디코더 개조. 원조 Chronos가 값을 이산 토큰으로 양자화해 한 스텝씩 자기회귀 생성했던 것과 달리, Bolt는 패치 단위 인코더 입력 + 디코더가 다중 스텝을 한 번에 직접 회귀 — 원조보다 수십 배 빠름 | `huggingface.co/amazon/chronos-bolt-*`, pip `chronos-forecasting` | 병렬 어텐션 + 사전학습(zero-shot) | 신규 도입 · 실행 환경 메인 venv · 적용 연구 27(예정) |
| TimesFM | Das 외(2024), *A decoder-only foundation model for time-series forecasting*, ICML 2024 | 디코더 전용 Transformer이지만 LLM처럼 한 스텝씩 생성하지 않고 각 디코딩 스텝에서 긴 출력 패치(예: 128스텝)를 한 번에 생성. 입력도 패치 토큰화해 가변 길이 입력·가변 예측 구간을 하나의 모델로 처리 | `github.com/google-research/timesfm`, pip `timesfm` | 병렬 어텐션 + 사전학습(zero-shot) | 신규 도입 · 실행 환경 메인 venv · 적용 연구 27(예정) |
| TTM (Tiny Time Mixers) | Ekambaram 외(2024), *Tiny Time Mixers (TTMs): Fast Pretrained Models for Enterprise Time-series Forecasting*, NeurIPS 2024 | 어텐션을 쓰지 않는 경량 MLP-Mixer 계열. 채널 간·패치 간 혼합을 전부 완전연결층으로 처리해 파라미터 수가 다른 파운데이션 모델 대비 수백 분의 1이고 추론이 빠름 | `github.com/ibm-granite/granite-tsfm`, pip `granite-tsfm` | 병렬 MLP-Mixer + 사전학습 | 신규 도입 · 실행 환경 메인 venv · 적용 연구 27(예정) |
| Moirai / Moirai-2.0 | Woo 외(2024), *Unified Training of Universal Time Series Forecasting Transformers*, ICML 2024 | 마스크드 인코더 Transformer. "any-variate attention"으로 다변량 수가 바뀌어도 같은 모델을 그대로 사용 가능, 여러 패치 크기에 대응하는 다중 입출력 프로젝션층으로 샘플링 주기가 다른 데이터를 함께 사전학습 | `github.com/SalesforceAIResearch/uni2ts`, pip `uni2ts`(torch 2.4.1+cu121 요구) | 병렬 어텐션 + 사전학습(zero-shot) | 신규 도입 · 실행 환경 격리 venv `.venvs/moirai_py312_20261004_192438` · 적용 연구 27(예정) |
| Sundial | *Sundial: A Family of Highly Capable Time Series Foundation Models*(Tsinghua THUML, 2025) | 자기회귀 디코더이지만 각 스텝에서 점 하나가 아니라 플로우 매칭(flow matching)으로 연속 확률분포 자체를 생성(TimeFlow loss). 패치 단위 입력, 네이티브 확률적 예측(여러 미래 경로 샘플링) 지원 | `github.com/thuml/Sundial`(`transformers==4.40.1` 고정 요구, 최신 transformers와 `DynamicCache` 인터페이스 비호환) | 어텐션 자기회귀 + 사전학습(zero-shot) | 신규 도입 · 실행 환경 격리 venv `.venvs/legacy_hf_py312_20261004` · 적용 연구 27(예정) |
| Time-MoE | *Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts*(2024) | 디코더 전용 Transformer에 희소 활성화 전문가 혼합(MoE) 층을 넣어, 추론 시 전체 파라미터 중 일부만 활성화하면서도 모델 총 용량은 키울 수 있게 함 | `github.com/Time-MoE/Time-MoE`(Sundial과 같은 이유로 `transformers==4.40.1` 고정 요구) | 어텐션 자기회귀(MoE) + 사전학습 | 신규 도입 · 실행 환경 격리 venv `.venvs/legacy_hf_py312_20261004`(Sundial과 공유) · 적용 연구 27(예정) |
| Lag-Llama | Rasul 외(2024), *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*, arXiv:2310.08278 | LLaMA 스타일 디코더 전용 Transformer를 원시 값이 아니라 lag(시차) 특징 위에서 학습시킨 최초의 공개 시계열 파운데이션 모델. 매 타임스텝마다 Student-T 분포의 모수를 출력하는 확률적 예측이며, 다중 스텝 예측을 한 스텝씩 순차 샘플링해 생성 — 이 때문에 패치 단위로 다중 스텝을 한 번에 내놓는 Moirai·Sundial·Time-MoE보다 처리량이 훨씬 낮음(구조적 특성, 결함 아님) | `github.com/time-series-foundation-models/lag-llama`(GluonTS `gluonts<=0.14.4` 의존, `pytorch_lightning` 경유로 구버전 `setuptools`[`pkg_resources` 포함 버전] 요구) | 어텐션 순차 샘플링 + 사전학습(zero-shot) | 신규 도입 · 실행 환경 격리 venv `.venvs/lagllama_py312_20261004` · 적용 연구 27(예정) |

---

## 7. 조사했으나 공식 구현 미확정 — 참고만 해 둔다

| 모델 | 처리 방식 | 문헌/비고(상태 · 근거) |
| :--- | :--- | :--- |
| GDN(Gated DeltaNet) | 선형어텐션(재귀형) | 미채택 · Yang·Kautz·Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025. 공식 구현(`github.com/NVlabs/GatedDeltaNet`)은 언어모델용 선형어텐션 빌딩블록이지, 시계열 예측 전용 공식 구현은 아직 없다 |
| TSMamba | 선택적 상태공간 + 사전학습 | 미채택 · *A Mamba Foundation Model for Time Series Forecasting*(arXiv:2411.02941). 유지보수 중인 공식 저장소를 특정하지 못함 |
| TimeFound | 사전학습 | 미채택 · 조사 당시(2026-10) 공식 저장소를 특정하지 못함 |
| MOMENT | 병렬 어텐션 + 사전학습 | 미채택 · 조사 당시(2026-10) 공식 저장소를 특정하지 못함(후속 재조사 필요) |

---

## 8. 계정·시스템 요구사항 때문에 실행이 막히는 모델

| 모델 | 요구사항 | 처리 방식 | 비고(상태 · 근거) |
| :--- | :--- | :--- | :--- |
| TimeGPT(Nixtla) | API 계정/키 필요 | 병렬 어텐션 + 사전학습(API) | 미채택 · 공개 가중치 없음, 공식 API 호출 전제 |
| xLSTM | CUDA 커널을 import 시점에 JIT 컴파일, `nvcc`(CUDA 컴파일러) 필요 | 순차·재귀 | 미채택 · 서버에 `nvcc` 미설치(시스템 레벨 설치라 `sudo` 필요), 메인 venv를 깨뜨린 사고 있었음(`uv remove xlstm`으로 복구) |

---

## 9. 초기 실험(1~20번, 방향예측 계열) 전용 — 현재 미사용

17번 이후 연구가 방향예측에서 변동성예측으로 전환되며 쓰이지 않게 된 모델들이다. 1~3번은
원 논문 이름을 그대로 딴 토이(toy) 구현이 많아, 원 아키텍처의 핵심 설계(아래 참고)를 거의
반영하지 않는다 — 코드 주석에도 "컨셉 간소화"·"not a full implementation"이라고 스스로
명시돼 있다. 재도입할 거라면 이 표의 "원 설계" 칸에 적힌 공식 구현을 새로 가져와야 한다
(자체 재구현 금지 원칙, AGENTS.md 2.13).

| 모델 | 원 논문(참고용) | 원 설계 | 이 저장소에서 실제로 한 것 | 처리 방식 | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ODE-RNN | Rubanova, Chen & Duvenaud(2019), *Latent ODEs for Irregularly-Sampled Time Series*, NeurIPS 2019 | 은닉 상태가 관측 사이 구간에서 신경 미분방정식(Neural ODE)을 따라 연속적으로 진화 — 적응형 ODE 솔버로 적분 | `ODERNNModel`(1·2·3번): GRUCell + 2층 MLP를 "0.1 스텝 Euler 적분"이라 부른 것 — 적응형 솔버 없음 | 순차·재귀 | 미채택 · 1~3번 이후 미사용 |
| mTAND | Shukla & Marlin(2021), *Multi-Time Attention Networks for Irregularly Sampled Time Series*, ICLR 2021 | 연속시간 임베딩을 여러 개의 시간-어텐션 커널로 학습해 불규칙 샘플링 시계열을 고정 격자로 재구성 | `mTANDModel`(2·3번): `nn.MultiheadAttention` 한 층 + 선형 시간 임베딩 | 병렬 어텐션 | 미채택 · 2~3번 이후 미사용 |
| Informer | Zhou 외(2021), *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021(Outstanding Paper) | ProbSparse self-attention으로 어텐션 복잡도를 O(L log L)로 줄이고, 증류(distilling)로 입력 길이를 단계적으로 압축 | `InformerModel`(2·3번): `TransformerModel`과 동일 코드에 dropout만 추가 — ProbSparse 어텐션 없음 | 병렬 어텐션(희소) | 미채택 · 2~3번 이후 미사용 |
| Non-stationary Transformer | Liu 외(2022), *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting*, NeurIPS 2022 | Series Stationarization(정규화) + De-stationary Attention(정규화로 지워진 고유 통계량을 어텐션에 다시 주입)의 두 모듈 조합 | `NonStatTFModel`(2·3번): 평균을 빼고 표준 Transformer를 태운 뒤 평균을 다시 더하는 것뿐 — De-stationary Attention 없음 | 병렬 어텐션 | 미채택 · 2~3번 이후 미사용 |
| N-BEATS(토이) | Oreshkin 외(2020), *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting*, ICLR 2020 | 이중 잔차(doubly residual) 스택 + 다항식/푸리에 기저 확장으로 추세·계절을 해석 가능하게 분해 | `NBeatsModel`(2·3번): 2층 MLP(`Linear→ReLU→Linear`) — 기저 확장·잔차 스택 없음 | 병렬 MLP 스택 | 미채택 · 재도입 시 공식 구현(`neuralforecast.models.NBEATS`/`NBEATSx`) 사용 |
| DeepAR(토이) | Salinas 외(2020), *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*, *International Journal of Forecasting* 36(3) | LSTM이 매 스텝 확률분포(보통 음이항·정규)의 모수를 출력, 학습 시엔 teacher forcing, 추론 시엔 샘플을 다시 입력으로 먹여 자기회귀 생성 | `DeepARModel`(2·3번): `LSTMModel`과 코드가 완전히 동일 — 확률분포 출력도 자기회귀 샘플링도 없음 | 순차·재귀 | 미채택 · 재도입 시 공식 구현(`neuralforecast.models.DeepAR`) 사용 |
| LinearDecomp | — (DLinear의 개념적 전신, 별도 논문 없음) | 추세·잔차로 분해 후 각각 선형층으로 예측 | `LinearDecompModel`(2·3번): 입력을 그대로 두 선형층에 각각 통과시켜 합산 — 실제 분해(이동평균 등) 없음 | 병렬 선형 | 미채택 · 이후 DLinear(§4, Zeng 외 2023)로 정식 대체 |
| LogisticRegression(방향 분류) | 표준 로지스틱 회귀 | 이진 분류 표준 모델 | `sklearn.linear_model.LogisticRegression`(20번): 방향(상승/하락) 분류 1회성 비교 | 특성 기반 분류(비신경) | 미채택 · 20번 이후 연구가 변동성예측(회귀)으로 전환되며 미사용. 유일하게 분류 문제였던 사례 |

같은 이름을 쓰는 TCN·Transformer(→VanillaTransformer)·PatchTST·Mamba·Autoformer도 1~3번에
토이 버전이 있었다(`ModelZoo` 클래스 내부, 2~3번) — 이것들은 새 행이 아니라 §4·§5의 해당
모델 행에 전신으로만 기록한다. 4번(텍스트-독립변수 분석)에도 LSTM·Transformer·Mamba·
TimeXer·iTransformer의 축소 프로토타입(`*Representative`/`*Lite`, 코드 자체가 "proxy, not
a full implementation"이라고 명시)이, 11번에는 Linear·PatchTST의 분포출력 버전
(`DistributionalLinear`/`DistributionalPatchTST`)이 있었다 — 전부 §2·§4의 해당 모델 변형이라
별도 행을 만들지 않는다.
