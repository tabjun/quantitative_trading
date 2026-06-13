# 주가·코인 시계열 예측 논문화 흐름과 최적화 안정화 필요성 문헌 리뷰

작성일: 2026-06-13  
목적: 3번 실험에서 Autoformer가 좋아 보였던 현상, 4/5/6번 실험의 필요성, 그리고 주가·코인 예측 논문들이 어떤 방식으로 방법론 비교와 최적화 문제를 처리하는지 정리한다.

## 1. 결론 먼저

현재 연구의 최종 목적은 단순히 “업비트나 주식 가격을 맞히는 모델 하나”를 만드는 것이 아니다. 논문화 가능한 연구로 만들려면 다음 질문에 답해야 한다.

- 비정상 금융 시계열에서 모델이 정말 미래 신호를 학습했는가?
- 아니면 직전가 복사, 0 수익률, 평균 수렴 같은 쉬운 해를 찾았는가?
- 독립변수, 텍스트, historical flow mart, 온체인/유동성 변수를 붙였을 때 개선 원인을 설명할 수 있는가?
- 단순 train loss가 아니라 out-of-sample, persistence baseline, 방향성, 거래비용, 민감도 분석까지 통과하는가?

최근 시계열 예측 논문들은 대체로 모델 구조 자체를 제안한다. 하지만 금융·코인 예측 논문은 모델 구조만으로는 부족해서, target 정의, leakage-safe split, normalization, validation-only tuning, baseline 비교, 손실함수, 거래비용, 민감도 분석을 같이 둔다. 즉 “최적화 문제”라는 제목을 달지 않더라도, 논문 안에서는 학습/검증/튜닝/평가 프로토콜로 반드시 다룬다.

따라서 현재 4/5/6번 작업은 논문 본문에서 다음 역할을 갖는다.

- `4번`: 독립변수 확장 후보가 실제 정보 가치를 갖는지 보는 단계
- `5번`: 5개 대표 구조와 target/loss 조합이 쉬운 해로 붕괴하는지 진단하는 단계
- `6번`: 독립변수 확장 전에 normalization, loss, target, model selection을 안정화하는 단계

이 흐름은 단순한 디버깅이 아니라, 논문에서 “방법론의 신뢰성”을 확보하는 핵심 절차다.

## 2. 3번 Autoformer 결과를 어떻게 이해해야 하는가

3번 실험에서는 15개 알고리즘을 비교했고, 그중 Autoformer 계열이 성능 수치상 좋아 보이는 구간이 있었다. 하지만 이후 보고서와 5번 진단에서 확인한 것처럼, 그 결과는 실제 예측력이 아니라 다음과 같은 착시였을 가능성이 크다.

- 가격 레벨이 자기상관이 매우 강해서, 다음 가격을 직전 가격처럼 내면 RMSE/MAE가 낮아 보인다.
- MinMax 스케일에서 1-step close를 직접 회귀하면 작은 변동을 못 잡아도 전체 가격선이 겹쳐 보인다.
- 예측 변화량이 실제 변화량보다 너무 작으면, 캔들 차트에서는 잘 맞아 보이지만 방향성과 수익률 예측력은 사라진다.
- 3번 보고서에서도 Autoformer·PatchTST 상위군은 “진짜 학습 성공”이 아니라 `평균 수렴형 과소적합` 또는 `지연 카피 과소적합`으로 해석해야 한다고 정리되어 있다.

즉 Autoformer가 나빴다는 뜻이 아니다. Autoformer는 decomposition과 auto-correlation을 통해 긴 시계열 벤치마크에서 강한 구조다. 문제는 우리 데이터와 평가 방식에서 “가격 레벨 복사형 착시”가 생길 수 있었다는 점이다. 그래서 5번에서 `level_mse`를 진단 대조군으로 두고, 6번에서 return target, normalization, loss ablation을 분리하는 것이 필요하다.

## 3. 최근 시계열 예측 알고리즘 흐름

아래 표는 다음 실험 후보군을 정리한 것이다. 중요한 점은 “Transformer 하나”가 아니라, decomposition, patching, variable-token, exogenous fusion, convolution, SSM/Mamba, foundation model, normalization까지 연구 흐름이 다양하다는 것이다.

| 계열 | 대표 논문 | 핵심 아이디어 | 우리 연구에 주는 의미 |
| --- | --- | --- | --- |
| Decomposition Transformer | Autoformer | trend/seasonal decomposition을 모델 내부 block으로 넣고 auto-correlation으로 sub-series dependency를 본다. | 3번의 Autoformer 착시는 모델 문제라기보다 target/evaluation 문제다. decomposition 구조는 6번 안정화 뒤 다시 검토 가능하다. |
| Frequency/decomposition | FEDformer | seasonal-trend decomposition과 Fourier/Wavelet frequency block을 결합한다. | 비정상 가격을 직접 회귀하기보다 주기/저주파/고주파 구조를 분리하는 방향으로 확장 가능하다. |
| Patch Transformer | PatchTST | 시계열을 한 점씩 보지 않고 patch 단위 token으로 만들어 attention 비용을 줄이고 긴 lookback을 본다. | 15분봉 고빈도 데이터에서 window를 길게 볼 때 OOM을 줄이는 후보가 될 수 있다. |
| Linear baseline critique | DLinear/LTSF-Linear | 단순 선형 모델이 복잡한 Transformer보다 잘할 수 있음을 보인다. | 우리도 복잡한 모델이 persistence/linear baseline을 이기지 못하면 채택하면 안 된다. |
| 2D temporal variation | TimesNet | 1D 시계열을 period 기반 2D tensor로 바꿔 intra/inter-period variation을 모델링한다. | 주기성이 약한 금융 데이터에서도 “복수 주기/장단기 변동” 진단 후보로 쓸 수 있다. |
| Inverted Transformer | iTransformer | timestamp token이 아니라 variable token에 attention을 적용한다. | 텍스트·유동성·온체인 등 독립변수가 늘어날 때 변수 간 관계를 보는 후보가 된다. |
| Exogenous Transformer | TimeXer | endogenous target과 exogenous variables를 patch-wise self-attention, variate-wise cross-attention으로 결합한다. | 4번 텍스트 독립변수와 historical flow mart를 붙일 때 가장 직접적인 구조 후보다. |
| Modern convolution | ModernTCN | 오래된 TCN을 현대화해 receptive field와 cross-variable dependency를 강화한다. | 5번에서 TCN이 상대적으로 덜 무너졌다면, ModernTCN 계열은 6번 이후 자연스러운 확장 후보다. |
| Selective SSM | Mamba / MambaTS | attention 없이 selective state space로 긴 sequence를 선형 복잡도로 처리한다. | 고빈도 15분봉에서 긴 lookback을 쓰되 GPU 메모리를 줄이는 후보가 된다. |
| Foundation model | Chronos, TimesFM, Moirai | 대규모 시계열 corpus로 사전학습해 zero-shot/few-shot forecasting을 시도한다. | 직접 학습이 불안정할 때 비교 기준 또는 teacher/reference forecast로 쓸 수 있다. 단 금융 특화 검증은 별도 필요하다. |
| Interpretable multi-horizon | TFT | static/known/observed covariates를 구분하고 variable selection/gating으로 해석 가능성을 높인다. | 교수님께 설명 가능한 독립변수 중요도와 multi-horizon 예측을 보여줄 때 후보가 된다. |
| Distribution shift normalization | RevIN | instance별 평균/분산 shift를 제거하고 출력에서 되돌린다. | 6번의 `window_standard`와 이후 RevIN layer 실험의 직접 근거다. |

## 4. 논문별 상세 정리

### 4.1 Autoformer

- 출처: Wu et al., “Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting”, NeurIPS 2021, https://arxiv.org/abs/2106.13008
- 무엇을 했나: 기존 Transformer가 긴 시계열의 복잡한 temporal pattern에서 reliable dependency를 찾기 어렵다고 보고, decomposition architecture와 Auto-Correlation mechanism을 제안했다. 논문은 long-term forecasting 벤치마크 6개에서 SOTA 개선을 보고한다.
- 학습/최적화 관점: 핵심은 모델 내부에 decomposition을 넣어 trend/seasonality를 점진적으로 분리하고, attention 대신 periodic sub-series dependency를 본다는 점이다.
- 결과 출력 방식: 여러 benchmark dataset에서 forecasting horizon별 MSE/MAE를 비교한다.
- 우리 연구 해석: Autoformer가 3번에서 좋아 보인 것은 Autoformer 자체가 허구라는 뜻이 아니다. 가격 레벨 직접 회귀와 MinMax 스케일에서 복사형 착시가 생긴 것이다. 따라서 Autoformer류는 6번에서 return target과 baseline-relative metric을 통과한 뒤 재평가해야 한다.

### 4.2 FEDformer

- 출처: Zhou et al., “FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting”, ICML 2022, https://arxiv.org/abs/2201.12740
- 무엇을 했나: Transformer가 전체 trend 같은 global view를 포착하지 못하고 계산량도 크다는 문제를 제기하고, seasonal-trend decomposition과 frequency-domain block을 결합했다.
- 학습/최적화 관점: 시간축을 그대로 다 맞히려 하지 않고 Fourier/Wavelet basis에서 sparse representation을 활용한다.
- 결과 출력 방식: multivariate/univariate long-term forecasting에서 prediction error 감소율을 제시한다.
- 우리 연구 해석: 업비트 가격 레벨은 비정상성과 급격한 regime shift가 강하므로 frequency decomposition은 후보가 될 수 있다. 다만 return target, volatility-scaled target과 같이 써야 복사형 착시를 피할 수 있다.

### 4.3 PatchTST

- 출처: Nie et al., “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers”, ICLR 2023, https://arxiv.org/abs/2211.14730
- 무엇을 했나: 시계열 한 시점씩 attention하는 대신 sub-series patch를 token으로 쓰고, channel-independence로 각 변수의 univariate sequence를 공유 encoder에 통과시킨다.
- 학습/최적화 관점: patching은 local context 보존, attention memory 감소, longer lookback 접근이라는 세 장점이 있다.
- 결과 출력 방식: long-term forecasting benchmark에서 SOTA Transformer와 비교하고, self-supervised pretraining/fine-tuning도 보여준다.
- 우리 연구 해석: OOM을 피하면서 긴 lookback을 쓰기 좋다. 그러나 3번에서 PatchTST도 좋아 보이는 착시가 있었으므로, 5/6번의 collapse 지표를 통과해야 한다.

### 4.4 DLinear / LTSF-Linear

- 출처: Zeng et al., “Are Transformers Effective for Time Series Forecasting?”, AAAI 2023, https://arxiv.org/abs/2205.13504
- 무엇을 했나: 복잡한 Transformer가 항상 필요한지 의문을 제기하고, 단순 one-layer linear 계열이 여러 LTSF benchmark에서 강하게 작동할 수 있음을 보였다.
- 학습/최적화 관점: 복잡한 구조보다 temporal relation extraction과 benchmark 설계가 더 중요할 수 있음을 보여준다.
- 결과 출력 방식: 9개 real-life dataset에서 Transformer 계열과 LTSF-Linear를 비교한다.
- 우리 연구 해석: 이 논문은 우리 연구의 안전장치와 맞닿아 있다. 딥러닝 모델이 persistence, linear, ARIMA, Random Forest 같은 단순 기준선을 못 이기면 논문에서는 “실패”가 아니라 “중요한 negative result”로 정리해야 한다.

### 4.5 TimesNet

- 출처: Wu et al., “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis”, ICLR 2023, https://arxiv.org/abs/2210.02186
- 무엇을 했나: 1D 시계열을 multiple periods 기반 2D tensor로 바꿔 intraperiod/interperiod variation을 2D kernel로 잡는다.
- 학습/최적화 관점: 모델이 시계열의 복잡한 변동을 한 줄짜리 sequence로만 보지 않게 한다.
- 결과 출력 방식: forecasting, imputation, classification, anomaly detection 등 5개 task에서 범용 backbone으로 성능을 비교한다.
- 우리 연구 해석: 15분봉에서는 일중/주중/이벤트 주기가 섞일 수 있다. 다만 crypto는 주기보다 shock이 강하므로, TimesNet은 historical flow나 regime feature와 같이 검토하는 것이 좋다.

### 4.6 iTransformer

- 출처: Liu et al., “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting”, ICLR 2024, https://arxiv.org/abs/2310.06625
- 무엇을 했나: 기존 Transformer는 timestamp token에 여러 변수를 섞어 넣기 때문에 변수별 의미가 흐려진다고 보고, time points가 아니라 variate token에 attention을 적용한다.
- 학습/최적화 관점: 변수 간 correlation을 attention이 직접 볼 수 있게 한다.
- 결과 출력 방식: 여러 real-world multivariate forecasting dataset에서 SOTA 성능과 generalization을 보고한다.
- 우리 연구 해석: 4번 이후 텍스트 독립변수, 유동성, 온체인, historical flow 변수를 늘릴 때 후보가 된다. 단, 변수가 늘어나면 노이즈도 늘어나므로 variable selection gate가 필요하다.

### 4.7 TimeXer

- 출처: Wang et al., “TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables”, NeurIPS 2024, https://arxiv.org/abs/2402.19072
- 무엇을 했나: target 자체(endogenous)만 보는 기존 방식이 실제 시스템에서는 부족하다고 보고, exogenous variables를 명시적으로 결합하는 Transformer를 제안했다.
- 학습/최적화 관점: patch-wise self-attention과 variate-wise cross-attention으로 target history와 외생변수를 연결한다.
- 결과 출력 방식: 12개 real-world benchmark에서 exogenous forecasting 성능을 비교한다.
- 우리 연구 해석: 4번 텍스트 독립변수와 historical flow mart의 장기 목적과 가장 가깝다. 하지만 바로 TimeXer로 가면 “변수 효과”와 “최적화 안정화 효과”가 섞인다. 그래서 6번이 먼저다.

### 4.8 ModernTCN

- 출처: Luo and Wang, “ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis”, ICLR 2024 Spotlight, https://openreview.net/forum?id=vpJMJerXHU
- 무엇을 했나: Transformer/MLP가 지배하는 흐름에서 convolution을 현대화해 time series에 다시 적용했다. 큰 effective receptive field와 cross-variable dependency를 강조한다.
- 학습/최적화 관점: attention보다 메모리와 속도 측면에서 유리할 수 있고, local/medium-range pattern에 강하다.
- 결과 출력 방식: forecasting, imputation, classification, anomaly detection 등 5개 task에서 SOTA와 비교한다.
- 우리 연구 해석: 5번에서 TCN 계열이 상대적으로 덜 무너진 점과 연결된다. 6번 통과 후 `TCN -> ModernTCN` 확장이 자연스럽다.

### 4.9 Mamba / MambaTS

- 출처: Gu and Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”, 2023/2024, https://arxiv.org/abs/2312.00752
- 출처: Cai et al., “MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting”, 2024, https://arxiv.org/abs/2405.16440
- 무엇을 했나: Mamba는 attention 없이 selective state space로 긴 sequence를 선형 복잡도로 처리한다. MambaTS는 이를 long-term time series forecasting에 맞게 변수 scan, temporal block, dropout, permutation training 등으로 수정한다.
- 학습/최적화 관점: 긴 context를 보면서 OOM을 줄이는 후보지만, 금융 시계열에서는 overfitting과 variable order sensitivity를 반드시 확인해야 한다.
- 결과 출력 방식: MambaTS는 public forecasting dataset 8개에서 성능을 보고한다.
- 우리 연구 해석: 서버 커널 제약과 긴 lookback 필요를 동시에 고려하면 후보가 된다. 다만 5번의 `MambaLite`처럼 proxy 구현만으로 성능을 주장하면 안 되고, 정식 구조/비교/ablation이 필요하다.

### 4.10 Chronos / TimesFM / Moirai

- 출처: Ansari et al., “Chronos: Learning the Language of Time Series”, 2024, https://arxiv.org/abs/2403.07815
- 출처: Das et al., “A decoder-only foundation model for time-series forecasting”, 2023/2024, https://arxiv.org/abs/2310.10688
- 출처: Woo et al., “Unified Training of Universal Time Series Forecasting Transformers”, 2024, https://arxiv.org/abs/2402.02592
- 무엇을 했나: 대규모 시계열 corpus로 사전학습한 foundation model을 사용해 zero-shot 또는 few-shot forecast를 시도한다.
- 학습/최적화 관점: 특정 작은 데이터셋에서 직접 학습이 불안정할 때, pretrained model을 baseline 또는 teacher처럼 사용할 수 있다.
- 결과 출력 방식: Chronos는 42개 dataset benchmark, Moirai는 LOTSA 27B observations, TimesFM은 다양한 public dataset zero-shot 성능을 보여준다.
- 우리 연구 해석: 업비트 15분봉 금융 데이터에서 바로 최종 모델로 쓰기보다, “우리 모델이 pretrained general forecaster보다 나은가?”를 보는 외부 기준선으로 쓰는 것이 안전하다.

### 4.11 Temporal Fusion Transformer

- 출처: Lim et al., “Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting”, International Journal of Forecasting, 2021, https://arxiv.org/abs/1912.09363
- 무엇을 했나: static covariates, known future inputs, observed historical inputs를 구분하고, gating과 variable selection으로 해석 가능한 multi-horizon forecast를 만든다.
- 학습/최적화 관점: black-box 예측이 아니라 어떤 변수가 언제 중요한지 보여준다.
- 결과 출력 방식: multi-horizon forecast 성능과 interpretability use case를 함께 제시한다.
- 우리 연구 해석: 교수님께 설명 가능한 독립변수 중요도와 예측 horizon별 시사점을 보여줄 때 강하다. 단, 먼저 6번에서 target/loss가 안정화되어야 변수 중요도 해석이 의미를 갖는다.

### 4.12 RevIN

- 출처: Kim et al., “Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift”, ICLR 2022, https://openreview.net/forum?id=cGDAkQo1C0p
- 무엇을 했나: 시계열의 평균과 분산이 시간에 따라 바뀌는 distribution shift 문제를 instance-level normalization과 denormalization으로 다룬다.
- 학습/최적화 관점: 입력에서 비정상적 평균/분산 정보를 잠시 제거하고, 출력에서 다시 복원한다.
- 결과 출력 방식: 여러 forecasting model에 RevIN을 붙여 성능 개선을 비교한다.
- 우리 연구 해석: 6번 `window_standard`는 RevIN의 완전 구현은 아니지만 같은 문제의 경량 진단이다. `window_standard`가 효과 있으면 이후 RevIN layer를 정식 도입할 근거가 된다.

### 4.13 Crypto price prediction review

- 출처: Wu et al., “Review of deep learning models for crypto price prediction: implementation and evaluation”, 2024, https://arxiv.org/html/2405.11431v1
- 무엇을 했나: cryptocurrency price forecasting 문헌을 리뷰하고, LSTM/CNN/Transformer 계열을 univariate/multivariate multi-step close-price prediction으로 비교했다.
- 학습/최적화 관점: 논문은 model training with Adam optimizer, data processing, framework, technical details를 별도 절로 둔다. 즉 최적화가 논문 밖의 일이 아니다.
- 결과 출력 방식: data analysis, pre-COVID/COVID-period scenario별 결과, volatility analysis, baseline 비교를 제시한다.
- 우리 연구 해석: 코인 예측 논문은 가격 예측만 말하는 것이 아니라 volatility regime과 dataset scenario를 함께 본다. 우리도 6번 이후 regime별 collapse/persistence 결과를 넣어야 한다.

### 4.14 Informer 기반 고빈도 Bitcoin 전략

- 출처: Stefaniuk and Ślepaczuk, “Informer In Algorithmic Investment Strategies on High Frequency Bitcoin Data”, 2025, https://arxiv.org/html/2503.18096v1
- 무엇을 했나: 5분/15분/30분 BTC 데이터를 대상으로 Informer를 RMSE, Quantile, GMADL loss로 학습하고, 예측값을 매매 전략으로 바꿔 Buy-and-Hold, MACD, RSI와 비교했다.
- 학습/최적화 관점: 이 논문은 loss function이 전략 성과에 어떤 영향을 주는지 연구 질문으로 직접 둔다. 또한 5/15분 데이터에서 RMSE Informer 예측이 거래비용보다 작은 0 근처로 나온 문제를 언급한다.
- 결과 출력 방식: portfolio value, annualized return, annualized standard deviation, information ratio, maximum drawdown, number of trades, long/short share를 window별로 비교한다.
- 우리 연구 해석: 이 논문은 현재 5/6번 문제와 거의 직접 연결된다. 고빈도 crypto에서는 RMSE로 학습하면 0 근처 예측이 나올 수 있고, 방향성/거래 목적에 맞춘 loss가 필요하다.

### 4.15 금융 시계열 모델 비교 논문

- 출처: Liu, “A Comparative Study of Transformer-Based and Classical Models for Financial Time-Series Forecasting”, Journal of Risk and Financial Management, 2026, https://www.mdpi.com/1911-8074/19/3/203
- 무엇을 했나: ARIMA, Random Forest, RNN, LSTM, CNN, Transformer를 6개 미국 주식 next-day log return 예측으로 비교했다.
- 학습/최적화 관점: walk-forward out-of-sample, validation-only hyperparameter selection, leakage control, train-only scaling, early stopping, fixed random seed를 명시한다.
- 결과 출력 방식: MAE/MSE/RMSE/Huber, heatmap, model correlation, SMA crossover benchmark, transaction cost 포함 결과를 보여준다.
- 우리 연구 해석: 이 논문은 “복잡한 deep model이 항상 이기지 않는다”는 결과도 출판 가능하다는 좋은 예다. 단, 엄격한 평가 설계와 baseline, tuning budget, leakage control이 있어야 한다.

## 5. 출판되는 논문들은 최적화 문제를 어디에 넣는가

질문은 “최적화가 중요한데 왜 논문 제목이나 초록에서 덜 보이는가?”이다. 답은 다음과 같다.

논문들은 최적화 문제를 대개 다음 위치에 넣는다.

- `Data and preprocessing`: log return, differencing, scaling, missing value 처리, train-only scaler
- `Experimental setup`: train/validation/test split, walk-forward validation, seed, hardware
- `Model training`: optimizer, learning rate, batch size, epoch, early stopping, gradient clipping
- `Hyperparameter tuning`: validation-only grid/random/Bayesian search, tuning budget
- `Loss function`: MSE, MAE, Huber, Quantile, direction-aware loss, trading objective loss
- `Evaluation`: RMSE/MAE/MAPE뿐 아니라 DA, hit ratio, MASE, Sharpe, MDD, transaction cost
- `Ablation/sensitivity`: lookback length, horizon, data frequency, feature set, loss, market regime
- `Baseline`: persistence, ARIMA, Random Forest, SMA, buy-and-hold, technical strategies

즉 논문이 “최적화 붕괴”라는 표현을 쓰지 않더라도, 학습 붕괴를 막기 위한 장치가 논문 방법론의 뼈대에 들어간다. 우리 연구는 현재 그 부분을 더 명시적으로 작성하고 있는 것이다.

## 6. 주가·코인 예측 논문의 일반적인 구성

출판되는 논문은 대체로 다음 흐름을 따른다.

### 6.1 연구 질문

- 특정 모델이 기존 모델보다 좋은가?
- 외생변수나 sentiment가 가격/수익률 예측을 개선하는가?
- 고빈도 데이터가 도움이 되는가?
- 손실함수나 target을 바꾸면 거래 성과가 좋아지는가?
- 복잡한 deep model이 classical baseline을 안정적으로 이기는가?

### 6.2 데이터

- 자산: Bitcoin, ETH, 주요 주식, 지수, 여러 종목 panel
- 주기: daily, hourly, 30분, 15분, 5분
- 입력: OHLCV, technical indicator, macro, sentiment, on-chain, cross-asset, order book
- 전처리: log return, rolling indicator, train-only scaling, missing value 처리

### 6.3 모델

- classical: ARIMA, GARCH, Random Forest, XGBoost
- recurrent: RNN, LSTM, GRU, BiLSTM
- convolution: CNN, TCN, ModernTCN
- transformer: Transformer, Informer, Autoformer, FEDformer, PatchTST, iTransformer, TimeXer
- foundation: Chronos, TimesFM, Moirai
- hybrid: CNN-LSTM, LSTM-XGBoost, sentiment + LSTM/Transformer

### 6.4 학습/최적화

- optimizer: Adam/AdamW가 많다.
- regularization: dropout, early stopping, weight decay, gradient clipping
- hyperparameter: lookback length, hidden dimension, batch size, learning rate, horizon
- split: walk-forward 또는 시간순 hold-out
- leakage control: train window 기준 scaler, validation-only tuning

### 6.5 결과 출력

- 예측오차: MAE, RMSE, MAPE, Huber
- 방향성: DA, hit ratio, F1, up/down/neutral classification
- baseline 대비: persistence, ARIMA, Random Forest, buy-and-hold, SMA/MACD/RSI
- 투자성과: cumulative return, Sharpe/Information ratio, MDD, turnover, transaction cost
- 그림: prediction vs actual, residual plot, loss curve, heatmap, regime별 성능, feature importance

## 7. 현재 연구에 바로 반영해야 할 기준

### 7.1 알고리즘 후보군

다음 단계에서 무작정 모델을 늘리면 3번의 문제를 반복한다. 따라서 6번 안정화 통과 후 아래 순서로 확장하는 것이 좋다.

| 우선순위 | 후보 | 이유 |
| --- | --- | --- |
| 1 | Persistence, Linear, DLinear, ARIMA/RandomForest | 논문에서 반드시 필요한 최소 기준선 |
| 2 | TCN, ModernTCN | 5번에서 TCN이 상대적으로 덜 무너졌고 메모리 효율도 좋음 |
| 3 | PatchTST | 긴 lookback/OOM 완화 후보 |
| 4 | iTransformer | 독립변수 증가 시 변수 간 관계 모델링 후보 |
| 5 | TimeXer/TFT | 텍스트·historical flow·외생변수 결합 후보 |
| 6 | Autoformer/FEDformer | decomposition/frequency 후보. 단 return target 기반 재검증 필수 |
| 7 | MambaTS | 긴 sequence와 OOM 제약을 동시에 다루는 후보 |
| 8 | Chronos/TimesFM/Moirai | 외부 pretrained 기준선 또는 zero-shot 비교 기준 |

### 7.2 평가 기준

보고서에는 최소 다음 기준을 같이 넣어야 한다.

- loss curve: train과 validation이 같이 내려가는가?
- persistence gap: 단순 직전가 복사보다 나은가?
- variance ratio: 예측 변동폭이 0으로 죽지 않는가?
- zero-share: 예측 수익률이 0 근처에 몰리지 않는가?
- sign agreement/DA: 방향성이 무작위보다 나은가?
- MASE: naive baseline 대비 상대 오차가 1 아래인가?
- MDD/transaction cost: 예측이 거래로 바뀌었을 때 방어적인가?
- regime split: 급등/급락/횡보 구간에서 성능이 다르게 무너지는가?

### 7.3 논문용 문장으로 정리할 핵심

현재 연구의 논문 방향은 다음처럼 잡는 것이 좋다.

> 본 연구는 비정상 고빈도 암호화폐 시계열에서 딥러닝 예측 모델이 단순 손실 감소에도 불구하고 persistence-copy 또는 zero-return shortcut으로 붕괴할 수 있음을 진단하고, 외생변수 기반 예측 모델로 확장하기 전에 target, normalization, loss, model-selection 기준을 안정화하는 절차를 제안한다. 이후 텍스트 독립변수와 historical flow data mart를 결합하여 예측 성능과 해석 가능성을 평가한다.

이 문장은 4/5/6번을 하나의 연구 흐름으로 묶는다.

- 4번은 외생변수 후보를 만든다.
- 5번은 모델이 쉬운 해로 무너지는지 보여준다.
- 6번은 학습 붕괴를 해소한 뒤 독립변수를 붙이기 위한 방법론 gate다.

## 8. 교수님께 설명할 때의 메시지

교수님께는 다음처럼 설명하는 것이 가장 자연스럽다.

> 최근 시계열 예측 논문들은 Autoformer, PatchTST, iTransformer, TimeXer, ModernTCN, Mamba, Chronos 같은 다양한 구조를 제안하지만, 금융·코인 예측 논문에서는 모델 구조보다 더 중요한 것이 leakage-safe split, target 정의, normalization, loss function, baseline 비교, transaction cost, sensitivity analysis입니다. 현재 연구에서는 3번에서 Autoformer가 좋아 보였던 현상이 실제 예측력이 아니라 복사형/평균수렴형 착시일 수 있음을 확인했기 때문에, 4번의 독립변수 확장으로 바로 넘어가지 않고 5번과 6번에서 학습 붕괴를 먼저 해소하고 있습니다. 이 과정을 통과해야 이후 텍스트 독립변수나 historical flow mart가 실제로 예측력을 개선했는지 논문으로 주장할 수 있습니다.

## 9. 다음 작업 제안

6번 서버 실행 후에는 아래 순서로 가는 것이 좋다.

1. `server_2048` 기준 6번 전체 stage 실행
2. 각 stage별 summary/curve CSV와 이미지 추출
3. 6번 결과 보고서 작성
4. 통과 후보 1~2개만 남겨 `7_model_family_expansion` 설계
5. `ModernTCN`, `PatchTST`, `iTransformer`, `TimeXer`, `MambaTS`, `Chronos/TimesFM/Moirai baseline` 중 OOM-safe 후보를 우선순위화
6. 4번 텍스트 독립변수와 historical flow mart를 붙인 exogenous forecasting 실험으로 확장

## 10. 참고 링크

- Autoformer: https://arxiv.org/abs/2106.13008
- FEDformer: https://arxiv.org/abs/2201.12740
- PatchTST: https://arxiv.org/abs/2211.14730
- DLinear / LTSF-Linear critique: https://arxiv.org/abs/2205.13504
- TimesNet: https://arxiv.org/abs/2210.02186
- iTransformer: https://arxiv.org/abs/2310.06625
- TimeXer: https://arxiv.org/abs/2402.19072
- ModernTCN: https://openreview.net/forum?id=vpJMJerXHU
- Mamba: https://arxiv.org/abs/2312.00752
- MambaTS: https://arxiv.org/abs/2405.16440
- Chronos: https://arxiv.org/abs/2403.07815
- TimesFM: https://arxiv.org/abs/2310.10688
- Moirai: https://arxiv.org/abs/2402.02592
- TFT: https://arxiv.org/abs/1912.09363
- RevIN: https://openreview.net/forum?id=cGDAkQo1C0p
- Crypto price prediction review: https://arxiv.org/html/2405.11431v1
- Informer high-frequency Bitcoin strategy: https://arxiv.org/html/2503.18096v1
- Financial model comparison with walk-forward validation: https://www.mdpi.com/1911-8074/19/3/203
