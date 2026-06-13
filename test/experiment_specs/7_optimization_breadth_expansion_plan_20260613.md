# 7. 넓은 폭 후속 확장 실험 계획서

작성일: 2026-06-13

## 1. 목적

7번은 5번 quick probe와 6번 안정화 오케스트레이터 사이에서 남은 공백을 메우기 위한 새 번호 실험이다. 5번은 대표 5개 구조를 약식으로 진단했고, 6번은 그 문제를 어떤 순서로 다시 볼지 계획했다. 7번은 이 둘을 이어, 더 넓은 모델군과 앙상블군을 포함한 후속 비교 실험으로 전체 흐름을 한 번에 파악하는 단계다.

이번 단계의 질문은 "5번에서 약식으로 본 collapse 문제가 더 넓은 모델 변형과 앙상블에서도 반복되는가"이다.

## 2. 왜 6번을 확장하지 않고 7번으로 분리하는가

- 6번은 이미 오케스트레이터라는 역할로 의미가 고정되었다.
- 6번을 넓은 후속 실험까지 떠안기면 기존 번호의 의미가 바뀐다.
- 연구 기록 추적성을 위해 후속 연구는 새 번호로 분리해야 한다.

즉 6번은 "무엇을 먼저 점검할지"를 정리한 문서와 실행 계획으로 남기고, 7번에서 실제 breadth expansion을 맡게 한다.

## 3. 모델군

### 3.1 기본 유지군

- `Linear`
- `LSTM`
- `GRU`
- `TCN`
- `Transformer`

### 3.2 확장 단일 모델군

- `Autoformer-like`
- `PatchTST-like`
- `DLinear-like`
- `NLinear-like`
- `TimesNet-like`
- `TimeXer-like`
- `iTransformer-like`
- `ModernTCN-like`
- `Mamba-like`

이번 단계에서의 `-like` 표기는 논문 완전 재현본이 아니라 representative family 구현이라는 뜻이다. 목표는 SOTA 재현보다 collapse/안정화 방향을 넓게 비교하는 것이다.

### 3.3 앙상블군

- `LSTM + Autoformer` soft ensemble
- `TCN + Transformer` soft ensemble
- `Linear + sequence residual stack`
- `validation weighted top-k ensemble`

앙상블군은 "auto-lstm 같은 혼합 모델도 함께 보고 싶다"는 연구 요청을 반영한다. 단일 모델에서 약하게 보였던 신호가 혼합 시 안정화되는지 확인하는 것이 목적이다.

## 4. suite 구성

### 4.1 `breadth_probe`

- 목적: 넓은 단일 모델군을 약식으로 한 번에 비교
- 범위: 기본 5개 + 확장 단일 모델군
- 결과: summary CSV, curve CSV, collapse figure, 모델별 상세 figure

### 4.2 `ensemble_probe`

- 목적: 혼합 앙상블군이 단일 모델보다 persistence baseline을 더 안정적으로 이기는지 확인
- 범위: 4개 앙상블군
- 결과: ensemble summary CSV, 단일 모델 대비 비교표

### 4.3 `normalization_cross_check`

- 목적: 6번에서 남긴 normalization 후보가 넓은 모델군에서도 같은 방향으로 유효한지 확인
- 범위: `standard`, `robust`, `window_standard`, 필요 시 `identity`

### 4.4 `loss_cross_check`

- 목적: 6번에서 남긴 loss 후보가 넓은 모델군과 앙상블군에서도 collapse를 줄이는지 확인
- 범위: `return_huber`, `directional_hybrid`, `volatility_weighted`, `tail_focus`

### 4.5 `scale_confirmation`

- 목적: 더 큰 `max_windows`, 더 긴 `epochs`, 보수적 batch로도 결론이 유지되는지 확인
- 범위: breadth_probe와 ensemble_probe의 상위 후보만 추려서 재검증

## 5. 지표와 해석 기준

7번도 이전 보고서를 참조하더라도 같은 설명을 다시 적어야 한다. 핵심 지표는 다음과 같다.

- `KRW MAE`, `RMSE`: 원화 스케일 오차
- `DA`: 방향 일치율
- `MASE`: naive baseline 대비 상대 오차
- `persistence_gap`: 0보다 아래여야 baseline 돌파
- `collapse_score`: flat/zero/copy shortcut 위험 보조 점수
- `variance_ratio`: 예측 변화폭 보존 여부
- `near_zero_return_share`: 0-return shortcut 비율
- `sign_agreement`: 방향성 우위 여부

좋은 그림은 다음 조건을 동시에 만족해야 한다.

- train/validation이 함께 하락
- gap이 과도하게 벌어지지 않음
- persistence_gap이 0 아래
- variance_ratio가 0에 붙지 않음
- zero_share가 과도하지 않음

## 6. 실행 원칙

- 실제 실행은 승인된 서버 환경에서만 수행한다.
- GPU stage는 순차 실행한다.
- OOM 발생 시 batch를 줄여 재시도한다.
- breadth 확장은 넓게 보되, 보고서는 반드시 케이스별 해석을 붙인다.

## 7. 산출물

- `test/models/7_optimization_breadth_expansion_test.ipynb`
- `test/models/7_optimization_breadth_expansion_test.py`
- `test/results/7_optimization_breadth_expansion_report_template_20260613.md`
- 후속 실제 실행 결과 CSV/PNG/Markdown

## 8. 보고서 작성 규칙

7번 보고서는 5번과 6번을 모른다고 가정하고 작성한다. 따라서 다음을 다시 적는다.

- 알고리즘이 무엇인지
- 앙상블이 무엇인지
- 왜 이번 단계에서 필요한지
- 지표와 그래프를 어떻게 읽는지
- 이번 결과가 무엇을 시사하는지

즉 7번 보고서는 독립적으로 보존되고 재활용될 수 있어야 한다.
