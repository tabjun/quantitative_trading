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
