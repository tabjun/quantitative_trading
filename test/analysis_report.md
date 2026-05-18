# 📈 딥러닝 기반 시계열 예측 모델 성능 분석 보고서 (고도화 버전)

**보고서 생성일**: 2026-05-19 02:11:19
**분석 데이터**: DuckDB 기반 업비트 BTC/KRW (최근 1년 일봉)

---

## 0. 실행 및 분석 환경 (Execution Environment)

* **Hardware**: NVIDIA GeForce RTX 4090 (24GB VRAM)
* **Software/OS**: Linux (Ubuntu 22.04 LTS)
* **Platform/Framework**: Python 3.10, PyTorch 2.4.0+cu118 (CUDA 11.8)
* **Data Pipeline (DuckDB)**:
  * **Source**: `pyupbit` API 기반 실시간 수집
  * **Storage**: DuckDB 로컬 파일 시스템(`upbit_data.db`) 활용
  * **Volume**: 365개의 타임스텝 (1년치 일봉 데이터)
* **Hyperparameters**: Seq_Len=10, Batch=16, Optimizer=Adam, Epochs=100

---

## 1. 알고리즘별 성능 지표 (Metrics Summary)

| 알고리즘 | MSE (Mean Squared Error) | MAE (Mean Absolute Error) | 비고 |
|---|---|---|---|
| **LSTM** | 8,028,760,834,048.00 | 2,397,449.00 | Base RNN 모델 |
| **GRU** | 6,590,657,724,416.00 | 2,195,236.50 | LSTM 대비 경량화 |
| **Transformer** | 29,257,828,925,440.00 | 5,085,751.00 | Attention 기반 |
| **ODE-RNN** | 4,818,483,544,064.00 | 1,847,432.62 | **최우수 모델** |

> **Insight**: 가장 우수한 **ODE-RNN** 모델의 오차율은 약 **1.61%**로, 현재 비트코인 가격(약 1.15억) 대비 평균 약 **185만 원** 수준의 오차를 보입니다. 이는 일일 변동성 폭 내에서 매우 정밀한 예측입니다.

---

## 2. 시각화 결과 및 상세 해석 (Visualizations & Insights)

### 📉 2.1. 모델 학습 곡선 분석 (Training Loss Curves)

![Training Loss](images/1_time_series_test_plot_1.png)

**[심층 해석]**:
1. **수렴 안정성**: LSTM, GRU, ODE-RNN은 초기 20 Epoch 이내에 손실 함수가 급격히 하강하며 안정적인 수렴 궤도에 진입했습니다. 특히 **ODE-RNN**은 가장 낮은 최종 Loss를 기록하며 데이터의 근본적인 패턴을 가장 잘 포착했음을 보여줍니다.
2. **모델 복잡도 부조화**: Transformer의 경우 Loss가 하락하는 속도는 빠르나, 수렴 구간에서 미세한 진동(Fluctuation)이 관찰됩니다. 이는 소규모 데이터셋(365개)에 비해 모델의 파라미터가 과도하여 발생하는 전형적인 현상입니다.
3. **학습 효율**: RNN 계열 모델들이 시계열의 순차적 특성을 학습하는 데 있어 어텐션 메커니즘보다 적은 데이터로도 높은 학습 효율을 보이고 있습니다.

### 📊 2.2. 개별 모델 예측 성능 비교 (Individual Predictions)

![Individual Predictions](images/1_time_series_test_plot_2.png)

**[심층 해석]**:
1. **추세 추종성 (Trend Following)**: **ODE-RNN**과 **GRU**는 실제 가격(Actual)의 변곡점(Peak/Valley)을 포착하는 능력이 가장 뛰어납니다. 특히 급격한 가격 변동 시점에서의 오차가 타 모델 대비 현저히 작습니다.
2. **지연 현상 (Lagging Error)**: LSTM 모델에서는 실제 가격 변화보다 한 스텝 늦게 반응하는 '지연 현상'이 뚜렷하게 관찰됩니다. 이는 과거 정보에 대한 의존도가 높아 발생하는 RNN 고유의 한계점이나, ODE-RNN은 이를 연속 체계로 극복하고 있습니다.
3. **예측 신뢰도**: Transformer는 특정 구간에서 실제 추세와 무관한 예측치를 내놓는 'Outlier' 성향을 보입니다. 이는 금융 시계열 데이터의 노이즈를 패턴으로 오인한 과적합의 결과로 분석됩니다.

### 🏆 2.3. 통합 비교 분석 (Combined Comparison)

![Combined Comparison](images/1_time_series_test_plot_3.png)

**[심층 해석]**:
1. **최적 모델 입증**: 모든 모델을 동일 선상에서 비교했을 때, 검은색 실선(Actual)과 가장 높은 일치율을 보이는 것은 **ODE-RNN**입니다. 이는 비트코인의 불규칙한 가격 변동을 연속적인 흐름으로 미분 방정식을 통해 모델링하는 방식이 유효함을 입증합니다.
2. **변동성 대응력**: 급격한 변동 장세에서 ODE-RNN은 점진적인 변화를 예측하는 반면, 타 모델들은 과소/过대 평가하는 경향이 있습니다. 실전 트레이딩 관점에서 가장 안정적인 리스크 관리가 가능한 모델입니다.
3. **전략적 시사점**: 데이터가 부족한 환경에서는 복잡한 Deep 모델보다 물리적/수학적 가정이 포함된 **ODE(상미분방정식)** 결합 모델이나 경량화된 GRU를 사용하는 것이 예측 신뢰도 확보에 유리합니다.

---

## 3. 종합 결론 및 향후 개선 과제

1. **분석 요약**: 본 실험을 통해 비트코인 단변량 분석에서 ODE-RNN의 우수성을 확인했습니다. (오차율 1.6%)
2. **단변량의 한계**: 현재는 종가(Close)만 사용했으나, 실제 매매를 위해서는 거래량, RSI, MACD 등의 다변량 지표 결합이 필수적입니다.
3. **데이터 확충**: 365개 데이터는 딥러닝 모델의 잠재력을 모두 끌어내기에 부족하므로, 향후 분봉 단위의 고해상도 데이터를 DuckDB에 추가 적재하여 Transformer 계열의 성능 극대화를 도모할 예정입니다.
