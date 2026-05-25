# 📓 Experiment Log: Advanced Time Series Analysis

본 문서는 시계열 가격 예측 알고리즘 분석 환경의 주요 변경점 및 실험 설계 고도화 이력을 기록합니다.

## 📅 2026-05-20: 전수 분석 매트릭스(Exhaustive Matrix) 도입

### 1. 주요 변경 사항
- **실험 자동화**: 기존 단일 모델 테스트 방식에서 벗어나, 전처리/모델/교차검증의 모든 조합을 자동으로 테스트하는 중첩 루프 구조 도입.
- **알고리즘 확장 (10종)**:
    - 기초: LSTM, GRU
    - 합성곱/분해: TCN, N-BEATS
    - 어텐션 계열: Transformer, Informer, Autoformer
    - 최신 SOTA: Mamba, mTAND, PatchTST
- **전처리 다양화**: `MinMaxScaler`와 `StandardScaler`를 병행 사용하여 스케일링 방식에 따른 모델 민감도 측정.
- **검증 강화**: 3-Fold TimeSeriesSplit을 통해 데이터 구간별 성능 편차 확인.

### 2. 실험 매트릭스 구조
| 구분 | 내용 |
|---|---|
| **데이터** | 업비트 BTC/KRW 15분봉 (3년치) |
| **전처리 (2)** | MinMax, Standard |
| **모델 (10)** | LSTM, GRU, TCN, N-BEATS, Transformer, Informer, Autoformer, Mamba, mTAND, PatchTST |
| **교차검증 (3)** | 3-Fold TimeSeriesSplit |
| **총 케이스** | **60 Cases (2 x 10 x 3)** |

### 3. 평가 및 시각화 고도화
- **학습 곡선 (Learning Curves)**: 모든 모델의 손실 함 수 추이를 저장하여 수렴 안정성 비교.
- **오차 분포 (Residual Analysis)**: 잔차 히스토그램을 통해 모델의 편향(Bias) 파악.
- **고급 지표**: DA(방향 정확도), MASE, DTW를 결합한 입체적 평가.
