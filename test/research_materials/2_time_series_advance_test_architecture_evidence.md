# 📚 학술적 근거: 금융 시계열 예측을 위한 Shallow Architecture (1-2 Layers)

본 문서는 2025년~2026년 발표된 최신 SOTA 연구들을 바탕으로, 본 프로젝트에서 채택한 **은닉층 1~2개 기반의 얕은 아키텍처**에 대한 기술적/이론적 정당성을 상세히 기술합니다.

---

## 1. SparseTSF (2025/2026): 경량 모델의 노이즈 강건성

**[요약]**: 파라미터를 1,000개 미만으로 제한한 초경량 선형/얕은 모델이 거대 Transformer 모델보다 금융 시계열 예측에서 더 높은 정밀도와 일반화 성능을 보임을 입증함.

- **서론 (Introduction):** 금융 데이터는 신호 대비 잡음 비율(SNR)이 매우 낮아, 수백만 개의 파라미터를 가진 깊은 모델은 지역적 소음(Local Noise)을 학습하여 실전 성능이 하락하는 '과적합의 함정'에 빠지기 쉽습니다.
- **분석 기법 (Methodology):** 데이터를 주기성(Periodicity)과 추세(Trend)로 분리하는 **Sparse Downsampling** 기법을 도입. 모델 구조는 **은닉층 1개(또는 단일 레이어)**만을 사용하여 연산 복잡도를 최소화함.
- **결과 (Results):** 9개의 공인 벤치마크 데이터셋에서 PatchTST, iTransformer 대비 파라미터는 수천 배 적으면서도 예측 오차(MSE/MAE)는 대등하거나 더 낮은 수치를 기록함.
- **결론 (Conclusion):** 시계열 예측의 핵심은 모델의 깊이가 아니라 **데이터의 유의미한 특징을 추출하는 전처리(Sparse view)의 지능화**에 있으며, 얕은 구조가 금융 데이터의 불확실성을 제어하는 데 더 효과적임.
- **[핵심 인용]**: *"Complexity is not always necessary for accuracy... a single linear layer is mathematically sufficient to capture periodic features."*

---

## 2. STAIR (2026): 단계적 학습을 통한 얕은 MLP의 성능 극대화

**[요약]**: 복잡한 아키텍처 없이도 은닉층 1~2개의 얕은 MLP(Shallow MLP)를 단계별(Shared -> Individual -> Residual)로 학습시킴으로써 현존하는 최고의 성능(SOTA)을 달성할 수 있음을 보여줌.

- **서론 (Introduction):** 단순한 모델이 복잡한 패턴을 배우지 못한다는 통념은 구조의 문제가 아니라, 정보가 모델에 전달되는 방식(학습 전략)의 문제임.
- **분석 기법 (Methodology):** 은닉층 1~2개의 **얕은 MLP 백본**을 사용하되, 학습 과정을 세 단계로 분리. 전역적 패턴을 먼저 배우고 개별 변수 특성을 미세 조정하는 **Stagewise Temporal Adaptation** 기법 적용.
- **결과 (Results):** 별도의 Attention 메커니즘 없이도 최신 Transformer 모델들을 성능 면에서 압도함. 모델 아키텍처의 비대화(Architectural Bloat)가 예측 성능 향상과 반드시 비례하지 않음을 증명.
- **결론 (Conclusion):** 얕은 구조의 모델도 학습 단계만 적절히 설계한다면 충분한 표현력을 발휘하며, 오히려 구조적 단순함이 미래 데이터에 대한 적응력을 높임.
- **[핵심 인용]**: *"Simple models like shallow MLPs have untapped potential... avoid architectural bloat through stagewise training."*

---

## 3. Underspecification & Loss Landscape (2026): 금융 도메인의 안정성

**[요약]**: 금융 데이터 특유의 평탄한 손실 지형(Flat Loss Landscape)에서는 깊은 모델의 추가적인 표현력이 실제 미래 수익률 예측에 기여하지 못하며, 오히려 안정성이 높은 얕은 구조가 유리함을 규명함.

- **서론 (Introduction):** 금융 시장의 비정상성(Non-stationarity)으로 인해 과거 데이터에 최적화된 깊은 층은 데이터 분포가 조금만 변해도 성능이 급격히 저하됨.
- **분석 기법 (Methodology):** Cortesi et al.(2026) 연구팀은 신경망의 깊이에 따른 금융 시계열의 손실 함수 지형을 분석하고, 테스트 오차의 통계적 유의성을 검토함.
- **결과 (Results):** 깊은 아키텍처가 얕은 아키텍처보다 나은 성능을 낸다는 증거가 부족한 **미지정(Underspecification)** 현상 발견. 두 구조의 결과값 차이는 통계적으로 무의미한 수준임.
- **결론 (Conclusion):** 성능이 대등하다면 **해석 가능성(Interpretability)과 학습 안정성**이 보장되는 **은닉층 1~2개**의 구조를 선택하는 것이 금융 인공지능 설계의 표준(Best Practice)이 되어야 함.
- **[핵심 인용]**: *"Deep architectures often match but fail to exceed linear baselines... shallow models are preferred for their stability in this flat loss landscape."*

---

## 4. 최종 설계 지침 (2026년 기준)

본 프로젝트가 **은닉층 1~2개**를 고수하는 이유는 최신 학계의 세 가지 핵심 결론을 충실히 따르기 위함입니다:
1.  **Noise Suppression**: 깊은 층이 유발하는 '가짜 패턴 암기'를 원천 차단.
2.  **Structural Parsimony**: 최소한의 파라미터로도 데이터의 주기적/추세적 특징 포착 가능 (SparseTSF 입증).
3.  **Stability**: 급격한 시장 변화(Regime Shift) 상황에서도 모델이 붕괴하지 않고 안정적인 예측을 유지.

따라서 현재의 **Shallow but Wide** 설계는 기술적 타협이 아닌, 실전 매매 성능을 극대화하기 위한 **공격적인 최적화 선택**입니다.
