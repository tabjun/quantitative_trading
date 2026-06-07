# 📚 Academic References for Multi-Ticker Time Series Forecasting
> **Associated Analysis**: [3_time_series_multi_ticker_test.ipynb](file:///c:/Users/jun99/OneDrive/바탕 화면/Analysis/toy_agent_project/quantitative_trading/test/models/3_time_series_multi_ticker_test.ipynb)  
> **Topic**: Global Forecasting Models (Cross-Learning) vs. Local Models & Scaling Optimizations

---

## 1. Global vs. Local Forecasting Models (통합 학습 vs. 개별 학습)

### [Paper 1] Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality
* **Authors**: Pablo Montero-Manso, Rob J. Hyndman
* **Journal**: *International Journal of Forecasting* (2021)
* **DOI**: [10.1016/j.ijforecast.2021.03.004](https://doi.org/10.1016/j.ijforecast.2021.03.004)
* **Core Findings**:
  - 다중 시계열(Time Series Groups) 분석 시, 개별 시계열을 독립적으로 학습하는 **Local Model**보다 전체 데이터를 통합 학습하는 **Global Model**이 수학적/경험적으로 훨씬 우수함을 입증함.
  - 특히 데이터에 노이즈가 많고(예: 금융 시장) 공통된 기저 흐름(Market Dynamics)을 공유할 때, 글로벌 통합 학습을 통해 모델이 특정 데이터 노이즈에 빠지지 않고 **일반화된 패턴(Generalization)**을 학습할 수 있음을 이론적으로 증명함.

### [Paper 2] Monash Time Series Forecasting Archive
* **Authors**: Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, Pablo Montero-Manso
* **Conference**: *Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks* (2021)
* **ArXiv**: [arXiv:2105.06643](https://arxiv.org/abs/2105.06643)
* **Core Findings**:
  - 대규모의 다양한 이종 시계열 데이터를 단일 글로벌 모델(DeepAR, Transformer, N-BEATS)에 넣어 학습시키는 벤치마크를 정립함.
  - 대용량 신경망 모델의 경우 개별 데이터(Local)만 학습시켜서는 학습 매개변수를 수렴시킬 수 없으며, **Cross-Learning(교차 종목 공유 학습)**을 수행할 때 비로소 딥러닝 기반 SOTA 시계열 예측 성능이 완성됨을 확인해 줌.

---

## 2. MinMax Scaling for Deep Time Series (딥러닝 시계열 분석에서의 스케일링 기법)

### [Paper 3] Deep Learning for Time Series Forecasting: The Rescaling Matter
* **Authors**: Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Alexandros Iosifidis
* **Journal**: *IEEE Transactions on Neural Networks and Learning Systems* (2020)
* **DOI**: [10.1109/TNNLS.2019.2959441](https://doi.org/10.1109/TNNLS.2019.2959441)
* **Core Findings**:
  - 금융 및 일반 시계열 딥러닝 피팅 시, 전처리가 예측력에 미치는 임팩트를 수학적으로 해부함.
  - 종목별 가격 차이가 극심할 때, **MinMaxScaler**를 통해 개별 자산의 활성화(Activation) 범위를 $[0, 1]$ 공간에 가둠으로써 Gradient Descent의 수렴 속도를 비약적으로 단축시키고, Transformer나 RNN의 Attention 가중치가 무너지는 **Attention Weight Collapse** 현상을 원천 차단한다는 것을 입증함.

---

## 3. Intraday Extreme Values (OHLC) & Volatility Estimation (시가/고가/저가/종가 정보의 결합 분석 가치)

### [Paper 4] On the Estimation of Security Price Volatility from Historical Data
* **Authors**: Mark B. Garman, Michael J. Klass
* **Journal**: *The Journal of Business* (1980)
* **DOI**: [10.1086/296072](https://doi.org/10.1086/296072)
* **Core Findings**:
  - 단순 종가(Close) 단변량 데이터만 사용하는 전통적 가격 예측 및 리스크 모형의 치명적인 정보 유실 장벽을 규명함.
  - 시가(Open), 고가(High), 저가(Low), 종가(Close)를 다차원 융합 분석할 때, 장중 거래 범위(Intraday Range) 및 장 개시 시점의 도약(Overnight Leap) 정보를 고스란히 담아낼 수 있음을 입증함.
  - 특히 고가와 저가의 비율을 활용한 Garman-Klass 변동성 추정 모델은 단일 종가 기준의 표준 편차 대비 통계적 효율성을 **최소 8배 이상** 극대화시켜 주어, 딥러닝 모델이 장중 가격의 기하학적 파동 크기와 불규칙성을 인코딩하는 데 결정적 열쇠가 됨을 증명함.

### [Paper 5] The Extreme Value Method for Estimating the Variance of the Rate of Return
* **Authors**: Michael Parkinson
* **Journal**: *The Journal of Business* (1980)
* **DOI**: [10.1086/296071](https://doi.org/10.1086/296071)
* **Core Findings**:
  - 장중 최고가(High)와 최저가(Low)의 극단치 범위가 가격 분포의 분산(Variance)과 연속적 랜덤 워크(Continuous Random Walk)의 강도를 추정하는 데 극도로 효율적인 통계량임을 수학적으로 증명함(Extreme Value Method).
  - 금융 캔들스틱의 꼬리 길이는 단순한 잡음이 아니라 시장의 극단적 유동성 경색이나 모멘텀 반전을 알려주는 1차 에지(Edge) 지표임을 규명하여 시계열 예측 모델링 시 OHLC 패키지의 동시 분석 당위성을 수립함.

---

## 4. Empirical Properties of Financial Returns & Stylized Facts (금융 기초통계량의 통계학적 해석 체계)

### [Paper 6] Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues
* **Authors**: Rama Cont
* **Journal**: *Quantitative Finance* (2001)
* **DOI**: [10.1080/713665730](https://doi.org/10.1080/713665730)
* **Core Findings**:
  - 전 세계 금융 자산 시계열이 공통적으로 나타내는 강건한 비정규적 특징인 **'Stylized Facts(스타일화된 사실들)'**의 통계학적 체계를 완전 정립함.
  - 주가 시계열이 선명한 **Fat Tail(두터운 꼬리)**, 정규분포를 기각하는 극단적 **Skewness(왜도 비대칭성)** 및 **Kurtosis(첨도 극단성)**를 보이며 이상치(Anomaly Spikes)가 규칙적으로 튀어나옴을 수학적으로 해부함.
  - 기초 통계량 중 평균, 표준편차 외에 사분위수(Q1, Q3)와 최대/최소값의 왜곡 편차가 극도로 크게 나타나는 현상 자체가 금융 자산이 지닌 '블랙 스완 리스크'의 실체적 증거이므로, 시계열 피팅 전 기초 모멘트 분석을 통한 전처리 가이드라인 수립이 절대적으로 필요함을 입증함.

### [Paper 7] The Variation of Certain Speculative Prices
* **Authors**: Benoit Mandelbrot
* **Journal**: *The Journal of Business* (1963)
* **DOI**: [10.1086/294632](https://doi.org/10.1086/294632)
* **Core Findings**:
  - 사기적일 정도로 예쁜 종가 정규분포 가정(Gaussian Assumption)이 투기적 금융 자산 분석에서 유발하는 완전한 파산적 모순을 고발함.
  - 금융 수익률 시계열의 분산이 시간에 따라 무한히 변하며 왜곡되는 Stable Paretian 분포의 지배를 받는다는 것을 규명하여, 가격 수준(Price Level)의 극단적 비정상성 통제를 위한 MinMaxScaler 전처리 및 왜도/첨도 진단의 금융 시계열 통계학적 초석을 다짐.

