# 모델 카탈로그 — 아키텍처 특징과 문헌 근거

이 문서는 **모델 자체의 특징과 문헌 근거만** 기록한다. 어느 실험(N번)이 이 모델을 실제로
채택했는지, 어떤 환경에서 돌렸는지, 왜 포함·제외했는지는 여기 적지 않는다 — 그건 각
실험의 `test/experiment_specs/{N}_*.md`(계획)와 `test/results/{N}_*/`(결과 보고서)가 다룬다.
이 문서는 실험 번호와 무관하게 누적되는 참조용 카탈로그라 번호를 붙이지 않는다(`test/
research_materials/`의 비번호 문서 관례 — `forecasting_methodology_literature_review_
20260613.md` 등과 같은 층위).

새 모델을 조사할 때마다 이 문서에 섹션을 추가한다. 기존 섹션은 내용이 틀렸을 때만 고치고,
"이번에 어디서 썼다" 같은 실험 서술은 넣지 않는다.

---

## 1. 통계·계량 모델

### HAR-RV (Heterogeneous Autoregressive model of Realized Volatility)
- 문헌: Corsi(2009), *A Simple Approximate Long-Memory Model of Realized Volatility*,
  *Journal of Financial Econometrics* 7(2).
- 특징: 로그 실현변동성을 과거 1봉·1일(평균)·1주(평균) 세 항의 선형회귀로 설명한다.
  변동성의 장기기억(long memory)을 복잡한 분수차분 없이 **이질적 투자기간을 가진 거래자가
  서로 다른 시간축을 본다**는 경제적 해석으로 근사한다. 파라미터가 3개뿐이라 과적합
  위험이 거의 없고, 변동성 예측 문헌에서 "이기기 어려운 표준 벤치마크"로 쓰인다.
- 구현: 선형회귀(자체 특성 생성 + `sklearn.linear_model.LinearRegression`) — 모델 자체가
  단순 선형식이라 자체 구현이 곧 공식 사양이다.

### GARCH-t (Bollerslev 1986/1987)
- 문헌: Bollerslev(1986), *Generalized Autoregressive Conditional Heteroskedasticity*,
  *Journal of Econometrics* 31(3); Bollerslev(1987)이 조건부 분포를 t분포로 확장.
- 특징: 조건부분산이 과거 조건부분산과 과거 제곱잔차의 선형결합으로 재귀적으로 갱신된다.
  조건부 분포를 정규분포 대신 **t분포**로 둬 금융수익률의 두꺼운 꼬리(fat tail)를 반영한다.
- 구현: `arch` 패키지(공식 유지보수 중인 GARCH 추정 라이브러리).

### MS-GARCH (Markov-Switching GARCH)
- 문헌: Hamilton(1989), *A New Approach to the Economic Analysis of Nonstationary Time
  Series and the Business Cycle*, *Econometrica* 57(2)의 마르코프 전환 필터를, Gray(1996)·
  Klaassen(2002)의 collapsing 근사로 GARCH 재귀에 결합한 구조.
- 특징: 국면(regime)이 **관측되지 않는** 잠재변수다. Hamilton 필터로 매 시점 국면확률을
  추정하고, Gray의 축약 절차로 국면경로가 2^t가지로 폭발하는 문제를 피한다. 두 국면 모두
  조건부밀도는 t분포(GARCH-t와 비교 가능하게).
- 구현: `engine/regime_garch.py`(자체 구현 — 경로 축약 근사 자체가 논문의 수식이라 패키지
  없이 직접 구현하는 것이 표준적인 방식).

### TAR-GARCH (Threshold/SETAR-GARCH)
- 문헌: 임계값 자기회귀(SETAR, Threshold Autoregression) 전통을 GARCH 조건부분산에 적용한
  레짐전환 GARCH 계열.
- 특징: MS-GARCH와 달리 국면이 **관측 가능한 임계변수**(예: 과거 수익률 부호·크기)가
  문턱값을 넘는지로 **즉시·결정론적으로** 전환된다. 필터링이 필요 없어 MS-GARCH보다
  계산이 단순하고 해석이 직관적이다.
- 구현: `engine/regime_garch.py`(자체 구현).

---

## 2. 선형·커널·트리 기반 지도학습

| 모델 | 핵심 아이디어 | 구현 |
| :--- | :--- | :--- |
| Linear / Ridge | 표준 (정규화) 선형회귀 | `sklearn.linear_model` |
| KernelRidge-RBF | RBF 커널로 비선형 관계를 커널 릿지로 학습 | `sklearn.kernel_ridge.KernelRidge` |
| SVR-RBF | RBF 커널 서포트벡터회귀 | `sklearn.svm.SVR`(libsvm) |
| Nystroem+Ridge | Nystroem 커널근사로 대표본에서 커널릿지를 선형 비용으로 근사 | `sklearn.kernel_approximation.Nystroem` + `Ridge` |
| LightGBM | 히스토그램 기반 그래디언트 부스팅(leaf-wise 성장) | 공식 `lightgbm` 패키지(Microsoft) |
| XGBoost | 그래디언트 부스팅(level-wise 성장 + 정교한 정규화) | 공식 `xgboost` 패키지 |
| HistGBM | 히스토그램 기반 그래디언트 부스팅(sklearn 내장) | `sklearn.ensemble.HistGradientBoostingRegressor` |
| GARCH+LightGBM | GARCH 조건부분산을 트리 특성으로 투입하는 하이브리드 | 위 두 구현의 조합(자체 결합 로직) |

이 계열은 아키텍처 자체가 잘 정립된 표준 알고리즘이라 원 논문 재현 이슈가 없다 — 전부
유지보수 중인 공식 패키지를 그대로 쓴다.

---

## 3. 순환신경망(RNN) 계열

### GRU (Gated Recurrent Unit)
- 문헌: Cho 외(2014), *Learning Phrase Representations using RNN Encoder-Decoder for
  Statistical Machine Translation*, EMNLP.
- 특징: LSTM보다 게이트 수가 적은(리셋·업데이트 게이트 2개) 경량 순환 구조. 장기 의존성을
  학습하면서도 파라미터가 LSTM보다 적다.
- 구현: `torch.nn.GRU`(PyTorch 공식 내장) 위에 우리 쪽 입출력 헤드를 얹은 구조.

### LSTM (Long Short-Term Memory)
- 문헌: Hochreiter & Schmidhuber(1997), *Long Short-Term Memory*, *Neural Computation* 9(8).
- 특징: 입력·망각·출력 3게이트와 별도의 cell state로 장기 의존성의 그래디언트 소실 문제를
  완화한 순환 구조. 시계열·언어모델링의 고전적 표준.
- 구현: `torch.nn.LSTM`(PyTorch 공식 내장).

---

## 4. Transformer·경량 혼합(Mixer) 계열 — neuralforecast 공식 구현

아래 전부 Nixtla의 `neuralforecast` 패키지(유지보수 중인 공식 오픈소스, pip 설치)가
원 논문 아키텍처를 구현해 제공한다. 자체 재구현(`engine/models.py`의 `~Like` 클래스들)은
AGENTS.md 2.13(모델명 Like 금지 — 검증된 구현 사용 의무)의 계기가 된 PatchTSTLike 채널혼합
오류 이후, 공식 구현이 있으면 그쪽을 쓰는 것으로 방향을 바꿨다.

### PatchTST
- 문헌: Nie 외(2023), *A Time Series is Worth 64 Words: Long-Term Forecasting with
  Transformers*, ICLR 2023.
- 특징: 시계열을 겹치는 패치로 나눠 토큰화하고(어텐션 연산량을 줄임), **채널독립
  (channel-independence)** — 다변량의 각 채널을 완전히 별도로 처리해 채널 간 거짓 상관을
  학습하지 않게 한다. 이 채널독립이 원 논문의 핵심 설계다.

### iTransformer
- 문헌: Liu 외(2024), *iTransformer: Inverted Transformers Are Effective for Time Series
  Forecasting*, ICLR 2024.
- 특징: 어텐션의 토큰 축을 "시간"이 아니라 "변수(채널)"로 뒤집는다 — 각 변수의 전체
  시계열을 하나의 토큰으로 취급해, 어텐션이 변수 간 관계를 학습하고 피드포워드가 시간
  축을 처리하게 한다.

### TCN (Temporal Convolutional Network)
- 문헌: Bai, Kolter & Koltun(2018), *An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling*, arXiv:1803.01271.
- 특징: 인과적(causal) 팽창 합성곱(dilated convolution)을 쌓아 수용영역을 지수적으로
  넓힌다. RNN의 순차 처리 병목 없이 병렬화가 가능하다.

### DLinear / NLinear
- 문헌: Zeng 외(2023), *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023.
- 특징: "복잡한 Transformer가 단순 선형모델을 못 이긴다"는 문제의식에서 나온 대조군.
  DLinear는 추세·계절 성분으로 분해 후 각각 선형층으로 예측, NLinear는 입력 마지막 값을
  빼는 정규화 후 선형층으로 예측(분포 shift에 강건).

### Autoformer
- 문헌: Wu 외(2021), *Autoformer: Decomposition Transformers with Auto-Correlation for
  Long-Term Series Forecasting*, NeurIPS 2021.
- 특징: 어텐션 대신 **자기상관(auto-correlation)** 메커니즘으로 주기적 의존성을 직접 찾고,
  내부에 추세-계절 분해 블록을 반복적으로 넣어 장기예측에서 분해와 예측을 함께 학습한다.

### TimesNet
- 문헌: Wu 외(2023), *TimesNet: Temporal 2D-Variation Modeling for General Time Series
  Analysis*, ICLR 2023.
- 특징: 1차원 시계열을 FFT로 찾은 주요 주기에 맞춰 2차원(주기×주기 내 위치) 텐서로 접어
  2D 합성곱으로 시간 내·시간 간 변동을 함께 포착한다.

### TimeXer
- 문헌: Wang 외(2024), *TimeXer: Empowering Transformers for Time Series Forecasting with
  Exogenous Variables*, NeurIPS 2024.
- 특징: 내생(예측 대상) 변수와 외생 변수를 별도 임베딩 경로로 분리해 처리한 뒤 교차
  어텐션으로 결합 — 외생변수가 있는 예측 문제에 특화됐다.

---

## 5. 합성곱·선형어텐션 계열 — 공식 GitHub 구현(neuralforecast 미제공)

### ModernTCN
- 문헌: Luo & Wang(2024), *ModernTCN: A Modern Pure Convolution Structure for General Time
  Series Analysis*, ICLR 2024 Spotlight.
- 특징: 어텐션 없이 순수 합성곱만으로 Transformer급 성능을 노린 구조. depthwise separable
  합성곱으로 시간·변수·특징 세 축을 독립적으로 확장(ConvNeXt의 시계열판에 가깝다).
- 공식 구현: `github.com/luodhhh/ModernTCN`(저자 공식, pip 패키지 없음).

### Mamba / S-Mamba (시계열 적용)
- 원형 문헌: Gu & Dao(2023), *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*, arXiv:2312.00752. Mamba 자체는 **범용 시퀀스 모델**이며 시계열·비정상성에
  특화된 구조가 아니다 — 언어·오디오·유전체 서열 등 긴 시퀀스를 어텐션의 O(n²) 대신
  선택적 상태공간모델(selective SSM)로 O(n)에 처리하는 것이 핵심 기여다.
- 시계열 적용 문헌: Wang 외(2024), *Is Mamba Effective for Time Series Forecasting?*,
  *Neurocomputing*(2024), 모델명 S-Mamba. 양방향(bidirectional) Mamba 블록으로 변수 간
  상관을, 피드포워드로 시간 의존성을 각각 처리한다. 긴 lookback을 선형 비용으로 봐서
  장기 의존성(변동성 지속성 등)을 포착하는 데 강점이 있다 — 단, MS-GARCH·TAR-GARCH처럼
  국면전환을 명시적으로 모델링하는 구조는 아니다.
- 공식 구현: `github.com/wzhwzhwzh0921/S-D-Mamba`(저자 공식, pip 패키지 없음).

---

## 6. 시계열 파운데이션 모델 (사전학습 + zero-shot)

언어모델처럼 대규모·다종 시계열로 사전학습한 뒤, 우리 데이터에 **재학습 없이(zero-shot)**
바로 추론한다는 점이 1~5절의 모든 모델과 다르다.

### Chronos-Bolt
- 문헌: Ansari 외(2024), *Chronos: Learning the Language of Time Series*, arXiv:2403.07815.
  Bolt는 그 후 공개된 고속 증류 버전.
- 특징: T5 기반 인코더-디코더를 개조. 원조 Chronos가 값을 이산 토큰으로 양자화해 한
  스텝씩 자기회귀 생성했던 것과 달리, Bolt는 패치 단위 인코더 입력 + 디코더가 **다중
  스텝을 한 번에 직접 회귀**하는 구조라 원조보다 수십 배 빠르다.
- 공식 구현: `huggingface.co/amazon/chronos-bolt-*`, pip `chronos-forecasting`.

### TimesFM
- 문헌: Das 외(2024), *A decoder-only foundation model for time-series forecasting*,
  ICML 2024.
- 특징: 디코더 전용 Transformer이지만 LLM처럼 한 스텝씩 생성하지 않고, 각 디코딩 스텝에서
  **긴 출력 패치**(예: 128스텝)를 한 번에 낸다. 입력도 패치 토큰화해 가변 길이 입력·
  가변 예측 구간을 하나의 모델로 처리한다.
- 공식 구현: `github.com/google-research/timesfm`, pip `timesfm`.

### TTM (Tiny Time Mixers)
- 문헌: Ekambaram 외(2024), *Tiny Time Mixers (TTMs): Fast Pretrained Models for Enterprise
  Time-series Forecasting*, NeurIPS 2024.
- 특징: 어텐션을 쓰지 않는 경량 MLP-Mixer 계열. 채널 간·패치 간 혼합을 전부 완전연결층
  으로 처리해 파라미터 수가 다른 파운데이션 모델 대비 수백 분의 1이고 추론이 빠르다.
- 공식 구현: `github.com/ibm-granite/granite-tsfm`, pip `granite-tsfm`.

### Moirai / Moirai-2.0
- 문헌: Woo 외(2024), *Unified Training of Universal Time Series Forecasting Transformers*,
  ICML 2024.
- 특징: 마스크드 인코더 Transformer. "any-variate attention"으로 다변량 수가 바뀌어도
  같은 모델을 그대로 쓸 수 있고, 여러 패치 크기에 대응하는 다중 입출력 프로젝션층으로
  샘플링 주기가 다른 데이터를 함께 사전학습했다.
- 공식 구현: `github.com/SalesforceAIResearch/uni2ts`, pip `uni2ts`. torch 2.4.1+cu121을
  요구(최신 torch와 호환성 확인 필요).

### Sundial
- 문헌: *Sundial: A Family of Highly Capable Time Series Foundation Models*(Tsinghua
  THUML, 2025).
- 특징: 자기회귀 디코더이지만 각 스텝에서 점 하나가 아니라 **플로우 매칭(flow matching)**
  으로 연속 확률분포 자체를 생성한다(TimeFlow loss). 패치 단위 입력, 네이티브 확률적
  예측(여러 미래 경로 샘플링) 지원.
- 공식 구현: `github.com/thuml/Sundial`. `transformers==4.40.1` 고정 요구(최신 transformers와
  `DynamicCache` 인터페이스 비호환).

### Time-MoE
- 문헌: *Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts*
  (2024).
- 특징: 디코더 전용 Transformer에 **희소 활성화 전문가 혼합(MoE)** 층을 넣어, 추론 시
  전체 파라미터 중 일부만 활성화하면서도 모델 총 용량은 키울 수 있게 했다.
- 공식 구현: `github.com/Time-MoE/Time-MoE`. Sundial과 같은 이유로 `transformers==4.40.1`
  고정 요구.

### Lag-Llama
- 문헌: Rasul 외(2024), *Lag-Llama: Towards Foundation Models for Probabilistic Time Series
  Forecasting*, arXiv:2310.08278.
- 특징: LLaMA 스타일 디코더 전용 Transformer를 원시 값이 아니라 **lag(시차) 특징** 위에서
  학습시킨 최초의 공개 시계열 파운데이션 모델. 매 타임스텝마다 Student-T 분포의 모수를
  출력하는 확률적 예측이며, 다중 스텝 예측을 **한 스텝씩 순차 샘플링**해 만든다 — 이
  때문에 패치 단위로 다중 스텝을 한 번에 내놓는 Moirai·Sundial·Time-MoE보다 처리량이
  훨씬 낮다(구조적 특성이지 결함이 아니다).
- 공식 구현: `github.com/time-series-foundation-models/lag-llama`. GluonTS(`gluonts<=0.14.4`)
  의존, `pytorch_lightning` 경유로 구버전 `setuptools`(`pkg_resources` 포함 버전) 요구.

---

## 7. 조사했으나 공식 구현 미확정 — 참고만 해 둔다

| 모델 | 비고 |
| :--- | :--- |
| GDN(Gated DeltaNet) | Yang·Kautz·Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025. 공식 구현(`github.com/NVlabs/GatedDeltaNet`)은 언어모델용 선형어텐션 빌딩블록이지, 시계열 예측 전용 공식 구현은 아직 없다 |
| TSMamba | *A Mamba Foundation Model for Time Series Forecasting*(arXiv:2411.02941). 유지보수 중인 공식 저장소를 특정하지 못함 |
| TimeFound | 조사 당시(2026-10) 공식 저장소를 특정하지 못함 |
| MOMENT | 조사 당시(2026-10) 공식 저장소를 특정하지 못함(후속 재조사 필요) |

---

## 8. 계정·시스템 요구사항 때문에 실행이 막히는 모델

| 모델 | 요구사항 | 비고 |
| :--- | :--- | :--- |
| TimeGPT(Nixtla) | API 계정/키 필요 | 공개 가중치 없음 — 공식 API 호출 전제 |
| xLSTM | CUDA 커널을 import 시점에 JIT 컴파일, `nvcc`(CUDA 컴파일러) 필요 | 서버에 `nvcc` 미설치(시스템 레벨 설치라 `sudo` 필요) |
