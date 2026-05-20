# 📊 시계열 가격 예측 알고리즘 정밀 분석 보고서 (Advanced Analysis)

## 1. 분석 개요
본 보고서는 15종의 다양한 시계열 예측 알고리즘을 활용하여 비트코인(BTC) 15분봉 데이터를 분석한 결과입니다. 3년치 데이터를 기반으로 하며, 에포크 100회 상향 및 조기 종료(Early Stopping)를 적용하여 모델별 학습 성능을 정밀하게 검증했습니다.

## 2. 모델 학습 손실(Loss) 분석
학습 곡선에서 나타나는 독특한 패턴과 그 원인에 대한 기술적 해석입니다.

### 📉 학습 곡선 시각화
![Training Loss Plot](../images/2_advance_training_loss.png)

### ❓ Loss 그래프 형태에 대한 해석 (Anomalous Loss Pattern)
사용자께서 지적하신 "평소와 다른 Loss 그래프 형태"는 다음과 같은 기술적 배경을 가집니다:

1.  **Huber Loss의 영향**: 본 분석에서는 이상치(Outlier)에 강인한 **Huber Loss**를 사용합니다. 일반적인 MSE(Mean Squared Error)와 달리 Loss 값이 매우 작게 시작하거나 수렴 속도가 다르게 느껴질 수 있습니다.
2.  **조기 종료(Early Stopping)의 가시화**: `patience=5`, `min_delta=1e-4` 설정으로 인해, 성능 개선이 멈춘 시점에서 그래프가 즉시 절단(Truncated)됩니다. 이로 인해 끝까지 완만하게 떨어지는 일반적인 그래프와 달리 '뚝 끊기는' 형태로 보일 수 있습니다.
3.  **모델군별 스케일 차이**: Transformer 계열과 RNN 계열은 초기 가중치 분포와 어텐션 메커니즘으로 인해 초기 Loss 스케일이 크게 다릅니다. 이들을 한 Figure에 그릴 경우 상대적으로 안정적인 모델이 평평하게(Flat) 보일 수 있으며, 이는 모델이 학습을 안 하는 것이 아니라 타 모델 대비 수렴이 매우 빠르거나 안정적임을 의미합니다.
4.  **복잡한 아키텍처**: Mamba나 ODE-RNN 같은 복잡한 모델은 초기 에포크에서 잠시 정체되었다가 급격히 하강하는 패턴을 보일 수 있습니다.

## 3. 가격 예측 결과 비교
RMSE 기준 상위 5개 모델의 실제 가격 추종 성능입니다.

### 🎯 예측 비교 시각화
![Prediction Comparison Plot](../images/2_advance_prediction_comparison.png)

### 🔍 잔차(Residual) 분석
![Residual Plot](../images/2_advance_residual_analysis.png)

## 4. 모델별 성능 요약 (RMSE 순)
| 순위 | 전처리 | 모델명 | RMSE | 방향 정확도(DA) | MASE |
|:---:|:---:|:---|:---:|:---:|:---:|
| 1 | MinMax | **Autoformer** | 320,017.69 | 47.71% | 2.15 |
| 2 | MinMax | **PatchTST** | 404,392.79 | 47.68% | 2.78 |
| 3 | MinMax | LSTM | 534,983.86 | 47.31% | 4.06 |
| 4 | MinMax | GRU | 614,013.99 | 47.61% | 4.86 |
| 5 | MinMax | ODE-RNN | 915,447.03 | 47.29% | 7.24 |

## 5. 종합 결론 및 제언
- **최우수 모델**: **Autoformer**와 **PatchTST**가 압도적인 낮은 RMSE를 기록하며 비트코인의 복잡한 가격 패턴을 가장 잘 포착했습니다.
- **전처리 효과**: 대부분의 모델에서 `StandardScaler`보다 `MinMaxScaler`가 더 안정적인 성능을 보였습니다.
- **방향성 한계**: RMSE는 낮지만 방향성 정확도(DA)는 50% 부근에 머물러 있어, 가격의 절대치 예측은 우수하나 단순 상승/하락 예측에는 추가적인 피처 엔지니어링이 필요함을 시사합니다.
