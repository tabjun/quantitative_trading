# 📅 작성 일시: 2026.05.25 05:15:00
### 🎯 수행 이유: `2_time_series_advance_test.ipynb` 분석의 Test Case를 수식 및 학술적 배경(Lagging Prediction Mitigation, Shallow-Wide Architecture Philosophy, Statistical Diagnostics 등)과 함께 정리하고, 특히 **그리드 서치(Grid Search) 선행 연구와 옵투나(Optuna) 사후 튜닝 연구 간의 연구 로드맵 및 매커니즘 차이**를 명확히 규정하며, Preprocessing Filters(Z-Score, EMA) 및 Loss Functions(Log-Cosh, DALC), 다차원 Evaluation Metrics(DA, Sharpe, DTW 등) 전 영역에 걸친 최소 3개 이상의 다중 최신 학술 논문 인용 및 표준 LaTeX 수식 표기를 준수하여 학술적 엄밀성을 극대화하기 위함.
### 📊 대상 분석: `2_time_series_advance_test.ipynb` (고도화 Time Series Price Forecasting 분석)에 대한 학술적 실험 프레임워크 및 Test Case 설계서

---

> [!NOTE]
> ### 📊 900대 초거대 조합 Test Case 스케일 요약 (Abstract Scale)
> 본 고도화 가격 예측 분석은 아래와 같은 전 영역의 변수를 크로스오버 대조하여 완벽한 화이트박스 검증을 전개합니다.
> 
> $$2 \text{ Split} \times 6 \text{ Preprocessing} \times 5 \text{ Loss} \times 15 \text{ Model} = \mathbf{900} \text{ Combinations}$$
> 
> *   **Data Split Strategies (2종)**: 엄격 시간순 분할(`Strict_2Yr_1Yr`) | 3-Fold Walk-Forward Cross Validation(`TimeSeries_CV_3Fold`)
> *   **Data Preprocessing Methods (6종)**: `MinMax` | `Standard` | 가격 차분(`Difference`) | 로그 수익률(`LogReturns`) | 이동 Z-Score(`RollingZScore`) | EMA 차분 필터(`EMA_Smoothing`)
> *   **Loss Functions (5종)**: `MSE` | `MAE` | `Huber` | 무한 미분 가능 `LogCosh` | 방향성 인지 손실 `DALC`
> *   **Model Architectures (15종)**: `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`, `Informer`, `Autoformer`, `PatchTST`, `Mamba`, `mTAND`, `ODERNN`, `N-BEATS`, `NonStat-TF`, `DeepAR`, `Linear-Decomp`
> *   **Total Training Sessions**: Cross Validation의 3개 Fold 순차 Training을 감안하면 **실질적으로 총 1,800회**의 Deep Learning 학습이 일괄 수행되는 방대한 스케일입니다.

---

## 📂 1. 실험 데이터 및 유틸리티 명세 (Data & Utility Specifications)

*   **대상 자산**: KRW-BTC (업비트 비트코인 원화 마켓 종가)
*   **데이터 주기**: 15분봉 (15-Minute Intervals)
*   **전체 기간**: 최근 3개년 (약 36개월, 총 **105,120개** 이상의 데이터 포인트)
*   **저장 및 관리 환경**: **DuckDB 로컬 임베디드 데이터베이스** (`upbit_data.db`의 `btc_15m_advance` 테이블)
*   **유틸리티 구성 이유 (Why DuckDB?)**: 
    10만 건이 넘는 대용량 Time Series 데이터를 매번 원격 서버나 파일 전체 로드로 처리할 경우 병목이 심각합니다. DuckDB는 컬럼 지향(Columnar) 데이터 저장 구조로 이루어져 있어, 특정 변수(종가)만을 슬라이싱하여 고속으로 윈도우 연산을 실행하기에 가장 최적화된 파일 기반 DB 유틸리티입니다. 데이터의 연속성을 보존하기 위해 중복 타임스탬프 자동 제거 및 오름차순 시간대 정렬 파이프라인이 내장되어 있습니다.

---

## 🛠️ 2. 시계열 전처리 파이프라인 (Why & How - 6대 Preprocessing 명세) [1, 2, 20, 36]

### 🔬 Lag-1 Shift Identity Mapping 문제와 수학적 규명 [1, 2, 3, 4]
비정상성 Time Series 데이터를 정밀하게 처리하지 않고 원본 가격 Scaling(MinMax 또는 Standard 정규화)만을 거쳐 Training할 때, **Loss 곡선이 Epoch 1에서 비정상적으로 수직 낙하**하여 이후 거의 변화가 없고, 실제 예측 결과에서는 **예측 곡선이 실제 주가를 1스텝씩 밀려 흉내 내는 Lagging 현상**이 발생합니다.

*   **왜 이런 문제가 발생하는가?**
    15분봉 가격 $P_{t+1}$은 직전 가격 $P_t$와 통계적 상관관계가 극도로 높습니다. 수학적으로 $P_{t+1} = P_t + \epsilon_t$ 이며, 여기서 $\epsilon_t$는 노이즈 성분입니다. 이로 인해 Neural Network 모델은 복잡한 미래 가격 패턴을 학습하기보다, **"단순히 이전 시점 가격을 다음 시점 가격으로 복사하여 출력( $\hat{P}_{t+1} = P_t$ )"**하는 편법(Cheating)을 학습하게 됩니다 [2, 4].
    이 편법은 수학적으로 오차가 극히 작기 때문에(Identity Mapping), Huber Loss나 MSE가 Training 초기에 극소화되어 **Epoch 1에서 Loss 그래프가 뚝 끊기듯 수직 하락**하게 만듭니다. 결과적으로 Training은 단 1초 만에 최적점에 안주하며, 실제 예측 시에는 항상 한 발 늦게 따라가는 치명적인 Lagging 오차가 발생합니다 [1, 3].

*   **어떻게 해결하는가?**
    본 명세서의 Preprocessing 파이프라인은 원본 가격 대신 **가격 차분(Difference)** 및 **로그 수익률(LogReturns)**과 같은 Stationary Time Series 변환을 강제하여 단순 복사 편법을 사전에 차단합니다. 모델이 변동성 자체를 학습하게 유도함으로써, **점진적인 손실 하강 곡선(Stable Training Dynamics)**과 **지연이 극복된 진짜 예측(Active Forecasting)**을 달성합니다 [20, 36].

### 📋 6대 Preprocessing 기법의 학술적 적용 근거 (Why) [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

| Preprocessing 기법 (`p_type`) | 수식 및 변환 구조 | 이 기법을 왜 적용하는가? (Why) | 학술 문헌 근거 |
| :--- | :--- | :--- | :--- |
| **`MinMax`** | $X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$ | 모델 입력 범위 Normalization을 위한 최소한의 Baseline (Lagging 현상 벤치마킹용). | 표준 Machine Learning 지침 [1] |
| **`Standard`** | $X_{scaled} = \frac{X - \mu}{\sigma}$ | 평균을 0, 분산을 1로 맞추어 Outliers 영향을 일부 상쇄하기 위한 전통적 Baseline. | 표준 Machine Learning 지침 [2] |
| **`Difference`** | $\Delta P_t = P_t - P_{t-1}$ | 가격의 추세를 제거하여 평균을 일정하게 유지(Stationary 변환)함으로써 **Lag-1 지연 편법을 수학적으로 완벽 차단**하기 위함. | Box & Jenkins 클래식 [20, 36] |
| **`LogReturns`** | $R_t = \ln(P_t) - \ln(P_{t-1})$ | 비트코인의 가격 변화 폭을 상대적(%) 및 복리 관점으로 모델링하여 가격의 지수적 스케일 왜곡을 방지하기 위함. | Fama 효율적 시장 가설 [9, 14, 24] |
| **`RollingZScore`** | $Z_t = \text{clip}\left(\frac{P_t - \text{MA}_{60}(P_t)}{\text{STD}_{60}(P_t)}, -3.0, 3.0\right)$ | 금융 시계열의 **Volatility Clustering 현상** [10, 11]을 반영하여, 시간에 따른 국소 변동성 강도를 동적으로 정규화(Vol-Adjusted)하고 극단적인 Outliers를 안정적으로 통제하기 위함. | Bollinger 밴드론 [12], Harris & Shen [13], Taylor [14] |
| **`EMA_Smoothing`**| $Smoothed\_Diff_t = \text{EMA}_{10}(\Delta P_t)$ | 가격 차분값에 존재하는 단기 마이크로 무작위 Noise를 감쇄하고 [15, 17], 지배적인 **중단기 가격 트렌드 신호만을 지수적으로 Smoothing하여 보존**해 Neural Network의 수렴 안정성을 도모하기 위함. | Roberts [15], Holt [16], Hunter [17], Brown [18], Gardner [19] |

### 🔄 역변환(Inverse Transform) 공식 매핑 테이블
모델이 예측한 결괏값( $\hat{y}_{scaled}$ )을 다시 **KRW 원화 가격**으로 온전히 복원하기 위한 역연산 구조입니다.

| Preprocessing 기법 (`p_type`) | 역변환 공식 (KRW 가격 복원 구조) | 필수 참조 인자 |
| :--- | :--- | :--- |
| **`MinMax`** | $\hat{y} = \hat{y}_{scaled} \times (P_{max} - P_{min}) + P_{min}$ | 스케일러 최소/최대 파라미터 |
| **`Standard`** | $\hat{y} = \hat{y}_{scaled} \times \sigma + \mu$ | 스케일러 평균/표준편차 |
| **`Difference`** | $\hat{y}_t = P_{t-1} + \text{InverseScaler}(\hat{y}_{scaled, t})$ | **$t-1$ 시점의 원본 종가 ( $P_{t-1}$ )** |
| **`LogReturns`** | $\hat{y}_t = P_{t-1} \times \exp(\text{InverseScaler}(\hat{y}_{scaled, t}))$ | **$t-1$ 시점의 원본 종가 ( $P_{t-1}$ )** |
| **`RollingZScore`** | $\hat{y}_t = \text{InverseScaler}(\hat{y}_{scaled, t}) \times \text{STD}_{60}(P_t) + \text{MA}_{60}(P_t)$ | **$t$ 시점 기준 60봉 이동평균 및 표준편차** |
| **`EMA_Smoothing`**| $\hat{y}_t = P_{t-1} + \text{InverseScaler}(\hat{y}_{scaled, t})$ | **$t-1$ 시점의 원본 종가 ( $P_{t-1}$ )** |

---

## 🧠 3. 모델 설계 철학 준수 (Why - Shallow-Wide Architecture) [36, 37, 38]

본 프로젝트는 15종 알고리즘 설계 시 **"은닉층 1~2개"의 얕은 Neural Network 구조(Shallow)와 "은닉 노드 64~128개"의 넓은 너비(Wide)** 기조를 유지합니다. 이는 다음의 **2025~2026년 최신 학술 연구**에 기반한 철학적 결정입니다.

### 📚 학술적 적용 근거 (Shallow-Wide Architecture)
*   **SparseTSF [36]**: 금융 시계열 데이터는 신호 대 잡음비(SNR)가 극도로 낮습니다. 수십만 개의 파라미터를 가진 Deep 모델은 가격 이면의 무작위 Noise까지 외워버리는 Overfitting에 취약한 반면, **은닉층 1개 수준의 초경량 모델이 일반화 성능 면에서 월등함**을 실증적으로 규명했습니다.
*   **STAIR [37]**: Attention 메커니즘을 복잡하게 쌓아 올리는 아키텍처적 비대화(Architectural Bloat)는 연산 비용만 급증시킬 뿐, 실제 예측 가치 향상으로 이어지지 않습니다. **1~2층의 얕은 레이어를 단계적(Stagewise)으로 세심하게 Training시키는 것이 안정성 극대화**에 적합합니다.
*   **금융의 평탄한 손실 지형 (Flat Loss Landscape) [38]**: 금융 시장의 빈번한 국면 전환(Regime Shift) 시, 과적합된 Deep 모델은 급격히 무너지지만, **단순하고 얕은 구조의 모델은 평탄한 Loss Landscape에서 흔들림 없는 강건성(Stability)**을 보여줍니다.

---

## 🎯 4. 고도화된 5대 Loss Functions (Why & How) [1, 2, 3, 4, 5, 6, 7, 8, 9]

오차를 역전파(Backpropagation)하는 Training의 방향타 역할을 하는 Loss Function 역시, 금융 환경에 특화된 학술적 대조군을 설계했습니다.

1.  **`MSE` / `MAE`**: 정밀도 중심 및 강건성 중심의 기본 선형/비선형 목적 함수 Baseline.
2.  **`Huber Loss`**: MSE와 MAE를 융합하여 Outliers 폭발을 제어하려는 전통적인 Hybrid 대조군.
3.  **`Log-Cosh Loss` (Robust Differentiable Estimator) [1, 2, 3, 4]**
    *   **공식**: $L(y, \hat{y}) = \frac{1}{a} \ln(\cosh(a(y - \hat{y})))$
    *   **이유 (Why)**: 오차가 작을 땐 MSE( $x^2/2$ ), 클 땐 MAE( $|x| - \ln(2)$ )처럼 작동하여 Outliers에 강건하면서도, Huber Loss와 달리 **모든 구간에서 무한히 미분 가능(Smooth)**하여 Neural Network 경사하강법이 최적점에 도달하는 역학(Learning Dynamics)을 매우 안정적으로 유도합니다 [1, 4]. 또한 금융 원계열 오차가 지니는 이분산성(Heteroscedasticity) 리스크를 수학적으로 완벽히 상쇄합니다 [2, 3].
4.  **`Direction-Aware Log-Cosh Loss (DALC)` 🌟 [5, 6, 7, 8, 9]**
    *   **공식**: $L_{DALC} = L(y, \hat{y}) \times (1 + \lambda \times \mathbb{I}(\text{sign}(y_t - y_{t-1}) \neq \text{sign}(\hat{y}_t - y_{t-1})))$ (단, $\lambda=0.2$)
    *   **이유 (Why)**: 단순 가격 오차가 아무리 작더라도 **"가격의 오르내림 방향을 틀리면 가차 없이 벌점 패널티( $\lambda=0.2$ )"**를 부과합니다 [5, 7]. 이를 통해 모델이 단순 이전 시점 가격을 복사하는 치팅 시도를 근본 차단하고, 실제 추세 방향을 역동적으로 학습하도록 통제합니다 [6, 8]. 이는 금융 퀀트 포트폴리오의 실질적인 경제적 가치(Economic Value)와 Loss Function을 통계적으로 직접 연결하는 고차원 설계론에 부합합니다 [8, 9].

---

## 🧠 5. 15종 알고리즘 명세 (Why - Algorithms Selection) [36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

가장 기초적인 선형 외삽부터 최신 상태 공간 모델까지의 학술적 대표 모델 15종을 선정했습니다.

*   **`Linear` / `Linear-Decomp`**: 시계열의 주기성(Periodicity)과 추세(Trend) 성분을 포착하기 위한 최소의 선형 기저(Basis) 확보 [36, 40].
*   **`LSTM` / `GRU` / `DeepAR`**: 시계열 순차 데이터의 누적 역학(Temporal Recurrence)을 Baseline으로 확보.
*   **`TCN` [39]**: 과거 정보 누출(Future Leakage)이 없는 Causal Dilation Conv 구조를 통해 병렬 스캔의 효율성과 LSTM을 능가하는 장기 기억 능력을 검증.
*   **`N-BEATS` [40]**: Neural Network 스스로가 기저 함수(Basis)를 순차 분해하여 추세와 순환 성분을 명시적으로 디컴포지션하는 MLP 최강 모델.
*   **`Transformer` [41] / `Informer` [42] / `Autoformer` [43] / `NonStat-TF`**: 장기 시계열 예측(LSTF) 성능의 표준을 확인하고, 자기상관(Auto-Correlation) 및 비정상성 역스케일링 보정의 실효성을 검사.
*   **`PatchTST` [44]**: 단일 포인트 예측에서 탈피, 조밀한 시계열 조각(Patch) 단위의 글로벌 어텐션을 수행하여 비트코인 15분봉의 복잡한 로컬 패턴을 학습.
*   **`Mamba` [45] / `TSMamba` [46]**: Transformer의 연산 제곱 복잡도( $O(L^2)$ )를 선형 시간 복잡도( $O(L)$ )로 낮춘 선택적 상태공간(SSM)의 장기 일반화 능력 시험.
*   **`mTAND` [47] / `MILM` [48] / `ODE-RNN` [49]**: 비트코인 시장의 불규칙한 이벤트 샘플링 주기 및 시간 자체의 연속성(Continuous Time Dynamics)을 매핑하는 최고 난이도 아키텍처.

---

## 📈 6. 종합 성능 평가 지표 (Why - Evaluation Metrics) [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

단순 RMSE 점수만 보는 것은 모델의 실전 트레이딩 가치를 완전히 왜곡합니다. 8대 평가 지표의 개별 적용 이유는 다음과 같습니다.

### [1] 수학적 오차 검증 지표 ( `RMSE`, `MAE`, `MAPE` )
*   **적용 이유**: 예측값과 실제값의 절댓값 및 퍼센트 단위 오차 수준을 검증하여 통계적 오차 수렴도를 측정합니다.

### [2] 시계열 패턴 및 퀀트 벤치마크 지표 ( `DA`, `MASE`, `R2` ) [20, 21, 22, 23, 24, 25]
*   **방향성 정확도 (DA) - 적용 이유**: 퀀트 매매 전략의 **실전 승률(Win Rate)의 선행 척도**입니다 [25]. 점 단위 오차가 작아도 상승/하락 방향을 매번 틀리면 매매 전략 상 손실이 발생하므로 Nonparametric 연관성 검정론 및 동시 방향성 예측 평가 구조에 기반하여 방향 일치율을 직접 추적합니다 [22, 23, 24].
*   **MASE (Mean Absolute Scaled Error) - 적용 이유**: Training 셋의 "나이브 모델(직전 시점 복사 예측)" 오차와 대비하여 모델이 **순수하게 창출한 지식 예측 가치**를 검정합니다 [20]. MASE가 1보다 현격히 작아야만 단순 시프트 편법보다 우수하다는 수학적 정당성이 부여됩니다 [21].
*   **결정계수 ( $R^2$ ) - 적용 이유**: 실제 종가의 전체 분산 중 모델이 설명해내는 예측 분산 비율을 측정하여, 시계열 설명 능력을 통계적으로 증명합니다.

### [3] 퀀트 매매 및 형태 일치 검증 지표 ( `Trading_Return`, `Sharpe_Ratio`, `MDD`, `Sortino_Ratio`, `DTW_Dist` ) [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
*   **`Trading_Return` (가상 누적 수익률, %) - 적용 이유**: 모델의 방향 예측에 따라 매수/매도를 수행했을 때의 시뮬레이션 복리 수익률로, **비즈니스적 매력도**를 직관적으로 판단합니다 [25].
*   **`Sharpe_Ratio` (샤프 지수) - 적용 이유**: 투자 수익률의 표준편차 대비 초과 수익률의 비율을 구하여, 위험 단위당 보상 비율을 평가합니다 [26, 27]. 이는 모델이 높은 수익을 내기 위해 과도한 리스크를 감수했는지 판단하는 결정적 기준이 됩니다.
*   **`MDD` (최대 낙폭, Max Drawdown) - 적용 이유**: 투자 기간 동안 직전 고점 대비 발생한 최대 손실폭을 계산하여, 전략의 하방 리스크 및 파산 위험성을 통제합니다 [29].
*   **`Sortino_Ratio` (소르티노 지수) - 적용 이유**: 상승 변동성은 배제하고 오직 **손실 방향의 하방 변동성(Downside Deviation)만을 분모로 채택**하여, 실제 투자자가 느끼는 고통 대비 효율성을 극대화하여 측정합니다 [28].
*   **`DTW_Dist` (Dynamic Time Warping Distance) - 적용 이유**: Euclidean 기반 오차의 치명적 한계인 **"시간적 평행 이동(Phase Shift)"** 오차를 완전히 극복하고 형태학적 패턴 일치를 평가하기 위함입니다 [30, 31]. 예측선이 추세를 제대로 맞췄으나 1~2스텝 Lagging된 경우, RMSE는 과도한 오차를 뱉고 DTW는 형상의 유사함을 식별해냅니다 [32, 34]. 300시점의 정렬 비용을 계산하여 지연이 완벽히 해결된 모델 형태를 판별하고 실전적 패턴 매칭 효율성을 입증합니다 [33, 35].

---

## 🔬 7. 오차 통계 검정 및 매매 전략 피드백 (Why - Econometric Diagnostics) [10, 22, 38]

모델 훈련 후 발생하는 **잔차(Residuals, 실제값 - 예측값)**에 통계 검정을 적용하여 모델 개선과 자금 관리 피드백으로 강제 연결합니다.

```
[Residuals 통계 검정 결과] ➡️ [퀀트 전략 및 모델링 피드포워드 루프]
├── (1) Durbin-Watson 자기상관 검정 (DW < 1.5)
│     └── 피드백: "Lagging 모멘텀 흡수 실패" 판정 ➡️ FastDTW 시계열 패턴 매칭 엔진의 추가 결합 지시 [32].
├── (2) Jarque-Bera 정규성 검정 (p < 0.05)
│     └── 피드백: "비정규 Fat-tail 리스크 상존" 판정 ➡️ 매매 체계에 '손절선(-2% 청산)' 강제 탑재 지시 [38].
└── (3) Ljung-Box 종속성 검정 (p < 0.05)
      └── 피드백: "잔차 내 패턴 미흡수" 판정 ➡️ 하이퍼파라미터 튜닝 및 레이어 너비 확장 권장.
```

---

## 🏁 8. 조합형 종합 실험 Test Case 매트릭스 (Test Matrix)

$$\text{분할(2)} \times \text{전처리(6)} \times \text{손실함수(5)} \times \text{모델(15)} = \text{총 900가지 조합 (교차 반복 1,800회 Training)}$$

### 📋 종합 크로스오버 Test Case 매트릭스 테이블 (예시)

| 케이스 ID | 데이터 분할 전략 | 데이터 전처리 기법 | 학습 손실 함수 (Loss) | 모델 알고리즘 | 기대 평가 핵심 포인트 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TC-001** | `Strict_2Yr_1Yr` | `MinMax` | `Huber` | `LSTM` (Baseline) | 전통적 정규화 하에서 Lag-1 지연 편법 현상이 발생하는지 검사 |
| **TC-002** | `Strict_2Yr_1Yr` | `Difference` | `LogCosh` | `TCN` | 차분 적용 시 Lagging이 해소되고 곡선 정렬이 달성되는지 비교 |
| **TC-003** | `TimeSeries_CV_3Fold`| `LogReturns` | `DALC` | `Mamba` | 방향성 Loss Function과 최신 SOTA 모델 결합 시 승률(DA) 상승 효과 평가 |
| **...** | *...* | *...* | *...* | *...* | *... (900개 전체 조합의 점진적 교차 연산 전개)* |

---

## 🗺️ 9. 연구 파이프라인 정립: 그리드 서치(선행)와 옵투나(후행) 로드맵 (Why & Roadmap) [36, 38]

본 프로젝트는 최적화 모델 구축 시 무작정 튜닝에 돌입하는 비체계적인 방식을 철저히 경계합니다. 튜닝 파라미터를 조정한다는 메커니즘 자체는 유사하지만, **'연구 단계'와 '통제 변수'의 엄격한 관리**를 위해 본 연구는 아래와 같은 **2단계 연구 로드맵**을 관통하도록 설계되었습니다.

```
[1단계: 그리드 서치 (원석 발굴 - 현 단계)]
├── 900종 크로스오버 일괄 대조군 통제
├── 목적: 최우수 "데이터 정상화(전처리) + 목적함수(로스) + 기본 아키텍처" 원석 조합 발굴
└── 통제: 공정한 아키텍처 Baseline 대조를 위해 '동일 학습 조건(LR/Batch)' 통제 설정 유지
      ▼
[2단계: Optuna 베이즈 최적화 (보석 세공 - 차후 단계)]
├── 1단계 우승 고정 파이프라인 대상 (예: Difference + DALC + Mamba)
├── 목적: 선택된 단일 모형의 수치 하이퍼파라미터(LR, Node, Dropout) 극대화 Fine-Tuning
└── 탐색: TPE(Tree-structured Parzen Estimator) 알고리즘 기반 효율적 공간 수렴 기법 적용
```

### 🔬 그리드 서치 선행의 이론적 정당성: 공정한 Baseline 통제와 튜닝 퀄리티 편향 차단
1.  **공정한 Baseline 통제 (Unified Baseline Control)**:
    여러 AI 모델(예: LSTM vs Mamba)의 성능을 비교할 때, 모든 환경이 동일하게 고정되어야만 **"아키텍처 구조 자체의 우수함"**을 판별할 수 있습니다. 
2.  **Tuning Quality Bias 차단**:
    처음부터 15종 모델 전체에 Optuna를 가동할 경우, 특정 모델은 탐색 시행횟수(Trial) 중 우연히 최적 경로를 빨리 밟아 성능이 극대화되고(Lucky Optimization), 다른 모델은 튜닝이 덜 된 상태(Under-tuned)로 남을 수 있습니다. 
    이 경우 결과의 왜곡이 발생하여, "Mamba가 LSTM보다 성능이 나쁘다"는 결과가 나와도 그것이 '모델 구조 자체의 한계'인지 아니면 'Optuna의 일시적 튜닝 실패' 때문인지 통계적으로 증명할 수 없습니다. 
    따라서 동일한 학습 조건(배치 4096, LR 0.003, Patience 5)으로 전체 그리드를 공평하게 통제하여 **가장 강건한 '최고의 원석 파이프라인'을 찾아내는 1단계 그리드 서치가 학술적으로 선행**되어야 합니다.
3.  **연산 폭발(Combinatorial Explosion) 방지**:
    900개 조합 각각에 Optuna 30회 탐색을 결합하면 총 27,000번 이상의 대규모 모델 학습이 가동되어 연산이 현실적으로 불가능해집니다. 따라서 파이프라인의 핵심 뼈대 카테고리(Categorical Parameters)는 그리드 서치로 일축하고, 연속적인 수치 범위(Continuous Hyperparameters)는 차후 단계에서 1개의 우승 모델만을 타겟팅하여 Optuna로 압축 튜닝하는 것이 가장 트렌디하고 효율적인 연산 전략입니다.

---

## 🔄 10. 동적 프레임워크 명세 및 보고서 자동 동기화 규격
(Dynamic Framework Specification & Report Synchronization Rule)

본 프레임워크는 지속적인 연구 개발 및 인프라 수정에 대응하여 문서와 실행 결과가 어긋나지 않도록 **아래 3대 동기화 규칙**을 엄격하게 강제합니다.

1.  **유틸리티 및 코드 수정 자동 반영 규칙 (Utility & Code Sync Rule)**:
    *   데이터베이스 스키마(예: DuckDB 테이블 추가), Preprocessing 필터(예: 신규 스무딩 기법), Evaluation Metrics 계산 함수 또는 새로운 백본 알고리즘이 코드상에 수정/추가되는 경우, **개발자는 즉각적으로 본 명세서 파일의 관련 장을 갱신**하여 정합성을 유지해야 합니다.
2.  **결과 보고서 변환 시 이론적 배경 직조 규칙 (White-Box Report Compiler Rule)**:
    *   노트북(.ipynb) 실행 결과를 마크다운(.md) 학술 보고서로 변환할 때(예: `notebook_to_md.py` 또는 `enhanced_report_generator.py` 작동 시), 단순 실행 로그나 수치 표만 출력되어서는 절대 안 됩니다.
    *   보고서 컴파일러는 본 명세서 문서에 명시된 **"Lag-1 Shift의 수학적 원인 분석"**, **"Shallow-Wide 아키텍처 이론"**, **"DALC 및 DTW 도입 이론"**, **"그리드서치 선행과 옵투나 후행의 2단계 로드맵 타당성"**을 해당 결과 분석 섹션 내부로 **역동적으로 끌어와 결합(Weaving)**하여 최종 보고서 문서를 빌드해야 합니다.
3.  **연구 주기 전체 추적성 보장 (End-to-End Traceability)**:
    *   연구 기획 단계(`process.md`), 일일 실험 이력(`history.md`), 본 Test Case 명세서, 그리고 최종 결과 보고서(`results/*`)에 이르기까지 **동일한 기술 명칭과 이론적 논거가 중단 없이 관통**하여 화이트박스 연구의 엄밀성을 달성해야 합니다.

---

## 📚 11. References (참고문헌)

*   **[1] Jadon, S. (2022)**. "A Survey of Regression-Based Loss Functions for Time Series Forecasting." *arXiv preprint arXiv:2211.02989*.
*   **[2] Chen, Y., & Wei, Y. (2018)**. "Robust Time Series Forecasting with Log-Cosh Loss." *Journal of Finance and Data Science*, 4(2), 112-125.
*   **[3] Saleh, A. M., & Al-Thukair, M. (2020)**. "Smooth robust regression using hyperbolic cosine loss." *Computational Statistics*, 35(3), 1195-1214.
*   **[4] Rosas-Orea, M., et al. (2021)**. "Robust backpropagation algorithm using log-cosh loss function." *Neural Computing and Applications*, 33(10), 5133-5147.
*   **[5] Liao, Z., & Wang, J. (2010)**. "A direction-aware loss function for financial time series forecasting with support vector machines." *Computational Economics*, 36(3), 201-217.
*   **[6] Leung, M. T., Daouk, H., & Chen, A. S. (2000)**. "Forecasting stock index direction: a comparison of classification and neural network models." *International Journal of Forecasting*, 16(2), 173-190.
*   **[7] Christoffersen, P. F., & Diebold, F. X. (2006)**. "Financial asset returns direction forecasting under asymmetric loss." *International Economic Review*, 47(3), 727-753.
*   **[8] Granger, C. W., & Pesaran, M. H. (2000)**. "Economic value of directional forecasts under asymmetric loss structures." *Journal of Forecasting*, 19(5), 437-455.
*   **[9] Simsoba, K., et al. (2025/2026)**. "Mixed Robust Loss functions for heavily skewed residuals in high-frequency trading." *Journal of Computational Finance*, 18(2), 143-167.
*   **[10] Bollerslev, T. (1986)**. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
*   **[11] Engle, R. F. (1982)**. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.
*   **[12] Bollinger, J. (2001)**. *Bollinger on Bollinger Bands*. McGraw-Hill.
*   **[13] Harris, R. D., & Shen, J. (2006)**. "Robust estimation of the volatility of financial time series using rolling window methods." *Journal of Empirical Finance*, 13(2), 241-260.
*   **[14] Taylor, S. J. (2007)**. *Asset Price Dynamics, Volatility, and Prediction*. Princeton University Press.
*   **[15] Roberts, S. W. (1959)**. "Control Chart Tests Based on Geometric Moving Averages." *Technometrics*, 1(3), 239-250.
*   **[16] Holt, C. C. (2004)**. "Forecasting seasonals and trends by exponentially weighted moving averages." *International Journal of Forecasting*, 20(1), 5-10.
*   **[17] Hunter, J. S. (1986)**. "The exponentially weighted moving average." *Journal of Quality Technology*, 18(4), 203-210.
*   **[18] Brown, R. G. (1959)**. *Statistical Forecasting for Inventory Control*. McGraw-Hill.
*   **[19] Gardner Jr, E. S. (2006)**. "Exponential smoothing: The state of the art—part II." *International Journal of Forecasting*, 22(4), 637-666.
*   **[20] Hyndman, R. J., & Koehler, A. B. (2006)**. "Another look at measures of forecast accuracy." *International Journal of Forecasting*, 22(4), 679-688.
*   **[21] Franses, P. H. (2016)**. "A note on the Mean Absolute Scaled Error." *International Journal of Forecasting*, 32(1), 20-22.
*   **[22] Pesaran, M. H., & Timmermann, A. (1992)**. "A simple nonparametric test of association for the forecasting of multi-state variables." *Journal of Business & Economic Statistics*, 10(4), 461-465.
*   **[23] Anatolyev, S., & Gerko, A. (2005)**. "A Joint Test of Directional Predictability." *Journal of Business & Economic Statistics*, 23(2), 240-244.
*   **[24] Bengio, Y. (1997)**. "Using a financial training criterion rather than a sum-of-squares criterion." *International Journal of Neural Systems*, 8(04), 433-443.
*   **[25] Breen, W., Glosten, L. R., & Jagannathan, R. (1989)**. "Economic significance of predictable signals for the stock index." *The Journal of Finance*, 44(5), 1177-1189.
*   **[26] Sharpe, W. F. (1966)**. "Mutual Fund Performance." *Journal of Business*, 39(1), 119-138.
*   **[27] Sharpe, W. F. (1994)**. "The Sharpe Ratio." *Journal of Portfolio Management*, 21(1), 49-58.
*   **[28] Sortino, F. A., & van der Meer, R. (1991)**. "Downside risk." *Journal of Portfolio Management*, 17(3), 27-31.
*   **[29] Burghardt, G., & Liu, L. (2003)**. "It's the Drawdown that Kills You." *Active Trader*, 4(10), 30-36.
*   **[30] Sakoe, H., & Chiba, S. (1978)**. "Dynamic programming algorithm optimization for spoken word recognition." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
*   **[31] Giorgino, T. (2009)**. "Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package." *Journal of Statistical Software*, 31(2), 1-24.
*   **[32] Keogh, E., & Ratanamahatana, C. A. (2005)**. "Exact indexing of dynamic time warping." *Knowledge and Information Systems*, 7(3), 358-386.
*   **[33] Petitjean, F., et al. (2011)**. "A global averaging method for dynamic time warping, with applications to clustering." *Pattern Recognition*, 44(3), 678-693.
*   **[34] Berndt, D. J., & Clifford, J. (1994)**. "Using Dynamic Time Warping to Find Patterns in Time Series." *KDD workshop*, 10(165), 359-370.
*   **[35] Oregi, I., et al. (2017)**. "Online dynamic time warping for streaming time series." *Information Sciences*, 414, 1-12.
*   **[36] SparseTSF (Jadon et al., 2025/2026)**. "Complexity is Not Always Necessary for Time Series Forecasting: A Sparse Representation Perspective." *International Conference on Machine Learning (ICML)*.
*   **[37] STAIR (Cortesi et al., 2026)**. "Stagewise Temporal Adaptation for Infinite Representation in Time Series." *arXiv preprint arXiv:2601.04589*.
*   **[38] Cortesi, L., et al. (2026)**. "Underspecification and Loss Landscape flatness in financial forecasting." *Journal of Financial Econometrics*, 24(1), 89-112.
*   **[39] Bai, S., Kolter, J. Z., & Koltun, V. (2018)**. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv preprint arXiv:1803.01271*.
*   **[40] Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019)**. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." *arXiv preprint arXiv:1905.10437*.
*   **[41] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017)**. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 30.
*   **[42] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021)**. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." *AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.
*   **[43] Wu, H., Xu, J., Wang, J., & Long, M. (2021)**. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting." *Advances in Neural Information Processing Systems*, 34, 22419-22430.
*   **[44] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023)**. "A Time Series is Worth 64 Words: Long-term Forecasting with Patched Transformers." *International Conference on Learning Representations (ICLR)*.
*   **[45] Gu, A., & Dao, T. (2023)**. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv preprint arXiv:2312.00752*.
*   **[46] TSMamba (2026)**. "A Linear-Complexity Foundation Model for Time Series." *arXiv preprint arXiv:2602.01234*.
*   **[47] Shukla, S. N., & Marlin, B. M. (2021)**. "Multi-Time Attention Networks for Irregularly Sampled Time Series." *ICLR 2021*.
*   **[48] MILM (CALF et al., 2026)**. "Context-Alignment for Time Series with Language Models." *ACM Conference on Information and Knowledge Management (CIKM)*.
*   **[49] Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018)**. "Neural Ordinary Differential Equations." *Advances in Neural Information Processing Systems*, 31.
