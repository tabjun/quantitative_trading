# 7. 넓은 폭 후속 확장 실험 보고서 템플릿

작성일: 2026-06-13

이 문서는 7번 실험 실제 실행 후 수치와 그림을 채우기 위한 독립 보고서 템플릿이다. 이전 보고서를 참고하더라도 핵심 설명을 생략하지 않는다.

## 1. 의도

- 왜 7번을 만들었는지
- 왜 6번을 덮어쓰지 않고 새 번호로 분리했는지
- 이번 단계에서 무엇을 넓게 확인하려는지

## 2. 설계

### 2.1 기본 유지군 설명

- `Linear`
- `LSTM`
- `GRU`
- `TCN`
- `Transformer`

각 모델이 무엇인지, 왜 이번 단계에서 비교군으로 유지하는지 다시 적는다.

### 2.2 확장 단일 모델군 설명

- `Autoformer-like`
- `PatchTST-like`
- `DLinear-like`
- `NLinear-like`
- `TimesNet-like`
- `TimeXer-like`
- `iTransformer-like`
- `ModernTCN-like`
- `Mamba-like`

각 모델군이 무엇인지, 어떤 구조적 차이를 보는지 다시 적는다.

### 2.3 앙상블군 설명

- `LSTM + Autoformer`
- `TCN + Transformer`
- `Linear + sequence residual stack`
- `validation weighted top-k ensemble`

앙상블의 의미와 기대 효과를 다시 적는다.

### 2.4 loss / normalization / 지표 설명

- `return_huber`
- `directional_hybrid`
- `volatility_weighted`
- `tail_focus`
- `standard`
- `robust`
- `window_standard`
- `KRW MAE`, `RMSE`, `DA`, `MASE`, `persistence_gap`, `collapse_score`, `variance_ratio`, `near_zero_return_share`, `sign_agreement`

## 3. 현재 결과

실제 실행 후 다음을 채운다.

- breadth_probe 요약
- ensemble_probe 요약
- normalization_cross_check 요약
- loss_cross_check 요약
- scale_confirmation 요약

## 4. 그래프 해석

### 4.1 좋은 그림 기준

- train/validation 동반 하락
- gap 과도 확대 없음
- persistence_gap 0 아래
- variance_ratio 과도 축소 없음
- zero_share 과도 집중 없음

### 4.2 이번 그림 해석

각 대표 그래프, collapse figure, case별 figure를 보고 실제 해석을 채운다.

## 5. 결과 해석

- 어떤 모델군이 덜 무너졌는지
- 어떤 normalization이 일관되게 나았는지
- 어떤 loss가 방향성과 baseline 돌파를 함께 만족했는지
- 앙상블이 실제로 개선을 만들었는지

## 6. 다음 스텝

- 후속 8번 또는 독립변수 확장으로 넘길지
- 아직 최적화 안정화가 더 필요한지
- 어떤 조합만 다음 단계로 남길지
