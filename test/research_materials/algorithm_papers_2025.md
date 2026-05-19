# 🧪 Test Environment Research Materials (2025-2026 Expansion)

본 폴더는 2025년 이후 발표된 최신 시계열 딥러닝 논문과 기술적 트렌드를 정리합니다. 특히 **Mamba(SSM)**와 **mTAND(Attention-based)** 아키텍처의 최신 진화 방향을 중심으로 분류했습니다.

---

## 1. Mamba (State Space Models) 계열: 선형 복잡도와 장기 의존성의 진화

Mamba는 2025년 이후 단순한 시퀀스 모델을 넘어, 파운데이션 모델(Foundation Models)과 하이브리드 아키텍처로 진화했습니다.

### [Foundation & Zero-Shot Forecasting]
- **TSMamba (2026)**: *A Linear-Complexity Foundation Model for Time Series*. 
  - **핵심**: LLM(Mamba-LLM)의 가중치를 전이 학습(Transfer Learning)하여 제로샷 성능 극대화. 양방향(Bidirectional) 인코더 도입.
- **Mamba4Cast (2025)**: *Efficient Zero-Shot Forecasting via SSMs*.
  - **핵심**: 수백만 개의 합성 시계열 데이터로 사전 학습된 파운데이션 모델. 트랜스포머 대비 10배 빠른 추론 속도.

### [Multi-Scale & Decomposition]
- **ms-Mamba (2025)**: *Multi-scale Mamba for Time-Series Forecasting*.
  - **핵심**: 서로 다른 샘플링 레이트($\Delta s$)를 가진 여러 Mamba 블록을 병렬 배치하여 시간적 해상도를 다각화.
- **DMamba (2025)**: *Decouple-and-Conquer Mamba*.
  - **핵심**: 시계열을 추세(Trend)와 계절성(Seasonality)으로 분해하고, 가변 방향 Mamba 인코더로 복잡한 계절성을 학습.

### [Hybrid Architectures]
- **SST (State Space Transformer, 2025)**: *Hybrid Experts for Long-Context Forecasting*.
  - **핵심**: Mamba로 장기 패턴을 스캔하고, Transformer 헤드로 단기 변동성을 정밀하게 보정하는 하이브리드 구조.

---

## 2. mTAND (Multi-Time Attention) 및 불규칙 시계열 계열

mTAND는 2025년 이후 LLM과의 결합 및 그래프 신경망(GNN)과의 융합을 통해 불규칙 샘플링 데이터 처리의 표준으로 자리 잡았습니다.

### [LLM & Multimodal Integration]
- **MILM (2026)**: *Multimodal Irregular time series Language Model*.
  - **핵심**: mTAND를 수치형 채널 인코더로 사용하여 텍스트 데이터와 결합. 불규칙한 임상/금융 데이터를 자연어 문맥으로 변환.
- **CALF (2026)**: *Context-Alignment for Time Series*.
  - **핵심**: mTAND 기반의 연속 시간 임베딩을 LLM의 프롬프트 컨텍스트와 정렬하여 예측 성능 향상.

### [Efficient & Structure-Aware Models]
- **IMTS-Mixer (2025)**: *MLP-Mixer for Irregular Time Series*.
  - **핵심**: mTAND의 어텐션 메커니즘을 보다 가벼운 MLP-Mixer 구조로 대체하여 파라미터 효율성 40% 개선.
- **WaveGNN (2026)**: *Decay-aware GNN for Irregular Sequences*.
  - **핵심**: mTAND의 다중 시간 보간(Interpolation) 기능을 그래프 구조 내의 감쇠 어텐션(Decay-aware)으로 확장.

### [Generative & Probabilistic]
- **iTimER (2025)**: *Uncertainty-aware Representation for Irregular Data*.
  - **핵심**: 단순 보간을 넘어 재구성 오차(Reconstruction Error)를 활용해 불규칙 시계열의 불확실성을 수치화.

---

## 3. 요약 및 시나리오별 모델 선택 가이드

| 시나리오 | 추천 모델 | 이유 |
| :--- | :--- | :--- |
| **초장기 컨텍스트 (LTSF)** | TSMamba, SST | $O(L)$의 선형 복잡도로 수만 스텝 참조 가능 |
| **불규칙한 샘플링 / 결측치** | mTAND, MILM | 연속 시간 임베딩을 통한 강력한 보간 능력 |
| **다양한 시간 주기 (다중 scale)** | ms-Mamba | 분/시간/일 단위의 복합 패턴 동시 학습 |
| **실시간 저지연 추론** | IMTS-Mixer, Mamba4Cast | 가벼운 파라미터와 병렬 스캔 메커니즘 |
| **불확실성 및 리스크 관리** | iTimER, Mamba-ProbTSF | 예측값뿐만 아니라 신뢰 구간(Variance) 동시 도출 |
