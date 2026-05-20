# 🧪 Test Environment Research Materials

이 폴더는 시계열 분석 고도화를 위해 참고한 최신 논문 및 기술적 배경지식을 정리하는 공간입니다.

## 1. 최신 시계열 모델 분석

### [Mamba: Selective State Space Models]
- **논문**: *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023/2024)
- **핵심**: Transformer의 $O(L^2)$ 복잡도를 $O(L)$로 개선하면서도 성능을 유지. 금융 시계열의 장기 의존성(Long-range dependency)을 효율적으로 압축.
- **적용**: 비트코인 3년치 15분봉 데이터의 장기 추세 파악.

### [mTAND: Multi-Time Attention Networks]
- **논문**: *Multi-Time Attention Networks for Irregularly Sampled Time Series* (ICLR 2021)
- **핵심**: 연속 시간 임베딩(Continuous Time Embedding)을 통해 불규칙하게 샘플링된 데이터나 결측치가 있는 시퀀스에서도 강건한 특징 추출.
- **적용**: 가상자산 시장의 변동성 및 불규칙한 거래 패턴 모델링.

### [Informer & Autoformer]
- **Informer**: ProbSparse Self-attention을 통한 연산 효율화 (AAAI 2021).
- **Autoformer**: 시계열 분해(Trend, Seasonality) 블록을 내장하여 주기적 패턴 학습에 특화 (NeurIPS 2021).

## 2. 실험 설계 (Experimental Design)

- **데이터 분할**: 2년(Train) / 1년(Test) Hold-out 검증.
- **교차 검증**: `TimeSeriesSplit`을 활용한 3-Fold Expanding Window 검증.
- **최적화**: `Optuna`를 활용한 하이퍼파라미터 베이지안 최적화.
- **손실 함수**: `MSE` (평균적 정확도) + `HuberLoss` (이상치 강건성).
