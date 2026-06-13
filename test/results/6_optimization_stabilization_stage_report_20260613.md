# 6. 최적화 안정화 오케스트레이터 결과 보고서

작성일: 2026-06-13

## 1. 의도

6번 실험의 목적은 5번 quick probe에서 드러난 shortcut collapse 문제를 바로 해결했다고 주장하는 것이 아니다. 이번 단계의 의도는 더 좁고 명확하다. 5번 결과를 바탕으로, 다음 후속 연구에서 무엇을 어떤 순서로 확인해야 하는지 실행 기준을 고정하는 것이다.

즉 6번은 "최종 성능 리더보드"가 아니라 "후속 안정화 실험의 오케스트레이터"다. 5번에서 확인한 문제는 다음과 같았다.

- 가격 레벨 직접 회귀는 직전가 복사형 쉬운 해에 빠지기 쉽다.
- return target으로 바꿔도 0 수익률 근처로 납작해지는 예측이 생길 수 있다.
- train loss가 줄어도 persistence baseline을 못 이기면 실전 예측으로는 실패다.

따라서 6번은 독립변수나 데이터마트를 바로 붙이기 전에, `target`, `normalization`, `loss`, `resource scale`, `gate 기준`을 어떤 순서로 다시 봐야 하는지 정리하는 단계로 설계했다.

## 2. 설계

### 2.1 이번 단계에서 사용한 방법론이 무엇인가

6번은 별도의 학습 엔진이 아니라, 기존 5번 진단 엔진을 순차적으로 호출하는 실행 계획 코드다.

- 기준 엔진: `test/models/5_optimization_diagnostics_test.ipynb`
- diff 추적 미러: `test/models/5_optimization_diagnostics_test.py`
- 오케스트레이터: `test/models/6_optimization_stabilization_test.ipynb`
- diff 추적 미러: `test/models/6_optimization_stabilization_test.py`

즉 실제 학습, 손실곡선 저장, CSV 저장, collapse 진단 그래프 생성은 5번 엔진이 담당하고, 6번은 그 5번 엔진을 어떤 순서와 조건으로 호출할지를 관리한다.

### 2.2 왜 오케스트레이터로 만들었는가

5번 결과에서 바로 "어떤 모델이 제일 좋다"를 고르면 위험했다. 현재 연구의 핵심 문제는 모델 서열보다 먼저, 손실함수와 목표 정의가 쉬운 해를 허용하는지 여부였기 때문이다. 그래서 6번은 한 번에 여러 축을 동시에 흔들지 않고 다음 순서로 한 축씩 확인하도록 설계했다.

1. 5번 결과 재현 확인
2. target/objective 재검토
3. normalization 비교
4. loss 비교
5. 창 수와 epoch를 키운 확장 확인
6. 독립변수 확장 전 gate 설정

### 2.3 이번 단계에서 다시 설명해야 하는 핵심 방법론

#### 대표 알고리즘 5개

- `Linear`: 가장 단순한 기준선이다. 구조적 표현력은 약하지만, 쉬운 해 여부를 보는 통제군으로 중요하다.
- `LSTM`: 과거 정보를 기억하는 recurrent 구조다. 시계열 학습에 널리 쓰이지만 flat prediction으로 수렴하는지 확인이 필요하다.
- `GRU`: LSTM보다 가볍고 학습이 빠르다. 같은 objective에서 더 안정적인지 비교하기 좋다.
- `TCN`: dilated convolution 기반 구조다. 국소 패턴을 빠르게 잡으면서 recurrent 계열보다 smoothing 방식이 다르다.
- `Transformer`: attention 기반 구조다. 복잡한 상호작용을 볼 수 있지만 복잡한 구조가 늘 좋은 것은 아니라는 점을 확인해야 한다.

#### 목표값과 손실함수

- `level_mse`: 다음 가격 수준을 직접 맞히는 MSE다. 가격 자기상관 때문에 직전가 복사형 해를 만들 위험이 크다.
- `return_huber`: 다음 로그수익률 또는 수익률 계열을 Huber loss로 학습한다. MSE보다 outlier에 덜 민감하다.
- `return_directional_hybrid`: 값 예측에 방향성 penalty를 얹는다. 0 수익률 shortcut이나 방향 무시형 답을 줄이려는 목적이다.
- `stabilization_loss_probe`: 6번 이후 단계에서 Huber, directional hybrid, volatility-weighted, tail-focus 계열 후보를 다시 확인하기 위한 비교 축이다.

#### 정규화 방식

- `standard`: 전체 기준 표준화다.
- `robust`: median과 IQR 중심으로 스케일 영향을 줄이는 방식이다.
- `window_standard`: 각 입력 window 내부 값만 사용해 평균/분산 shift를 완화하는 경량 방식이다.
- `identity`: 정규화 없이 그대로 두는 통제군이다.

#### 핵심 성능 및 붕괴 진단 지표

- `KRW MAE / RMSE`: 최종적으로 원화 스케일에서 얼마나 틀렸는지 본다.
- `DA`: 방향을 얼마나 맞히는지 본다.
- `MASE`: naive baseline 대비 상대 오차를 본다.
- `persistence_gap`: `모델 MAE - 직전가 복사 baseline MAE`다. 0보다 작아야 baseline을 이긴다.
- `collapse_score`: flat prediction, 0-return shortcut, copy risk를 묶어 보는 보조 점수다.
- `variance_ratio`: 예측 변화폭이 실제 변화폭을 얼마나 따라가는지 본다. 0에 너무 가까우면 평평한 예측이다.
- `near_zero_return_share`: 예측 수익률이 0 근처에 몰린 비율이다.
- `sign_agreement`: 방향성 일치율이다. 0.5 부근이면 무작위 수준이다.

### 2.4 그래프를 어떻게 읽어야 하는가

6번은 실제 그래프 묶음을 다시 생성한 단계가 아니라, 이후 그래프를 어떻게 읽을지를 명시하는 단계였다. 그럼에도 보고서는 그래프 해석 기준을 다시 적어야 한다.

- `Train/Validation Loss`: 둘이 함께 내려가야 한다. train만 내려가면 외우기일 수 있다.
- `Validation - Train Gap`: 0보다 위로 벌어지면 일반화가 약해진다.
- `Gradient Norm`: 너무 크면 불안정, 너무 작으면 학습 신호가 약하다.
- `Persistence Gap`: 0보다 아래여야 naive copy보다 낫다.
- `Collapse Diagnostics`: loss가 줄더라도 variance_ratio, zero_share, sign_agreement가 나쁘면 여전히 shortcut 학습이다.

## 3. 현재 결과

### 3.1 현재 저장된 6번 출력은 무엇인가

현재 `test/models/6_optimization_stabilization_test.ipynb`에 저장된 출력은 대규모 학습 결과 묶음이 아니다. 저장된 출력은 다음을 보여주는 stage plan이다.

- `profile = server_2048`
- Stage 0: 5번 quick probe 재현
- Stage 1: target/objective gate
- Stage 2: normalization ablation
- Stage 3: loss ablation
- Stage 4: resource scale check
- Stage 5: independent variable gate

즉 현재 6번 결과는 "무슨 순서로 추가 연구를 할지"를 보여주는 실행 계획이며, 각 stage의 목적, 좋은 그림 기준, 실패 신호, 판단 규칙이 텍스트로 저장되어 있다.

### 3.2 왜 이것을 결과로 보존해야 하는가

이번 출력은 숫자 성능표는 아니지만 연구 흐름상 중요한 결과다. 이유는 다음과 같다.

- 5번 이후 무엇을 먼저 확인해야 하는지 연구 순서를 고정한다.
- 후속 실험에서 어떤 결과가 나오면 통과이고, 어떤 결과가 나오면 재설계인지 기준을 남긴다.
- 이후 7번처럼 넓은 폭의 실험을 만들 때도, 6번의 gate 기준이 기준점이 된다.

즉 이번 단계의 결과는 "학습 성능"이 아니라 "연구 의사결정 규칙"이다.

## 4. 결과 해석

### 4.1 의도대로 나온 부분

현재 6번 출력은 오케스트레이터라는 작성 의도에는 맞게 나왔다.

- 기본 실행이 곧바로 학습을 돌리지 않고 stage plan을 출력한다.
- 각 stage마다 무엇을 바꾸는지, 왜 지금 그 축을 보는지, 어떤 그림이 좋아야 하는지 설명한다.
- independent variable 확장 전에 최소 안정성 gate를 둔다.

이 점에서 6번은 "5번의 문제를 임시 메모로 남긴 것"이 아니라, 후속 실험 순서를 정리한 구조화된 실행 계획으로 기능한다.

### 4.2 의도대로 나오지 않은 부분

반대로, 사용자가 originally 기대한 "테스트 케이스를 더 늘리고 실제 결과를 많이 쌓아서 방향을 잡는 단계"로는 아직 부족하다.

- 현재 출력은 실제 학습 곡선 비교 묶음이 아니다.
- 더 넓은 모델군과 앙상블군 결과가 아직 없다.
- normalization/loss를 실제로 돌린 뒤의 CSV, PNG, Markdown 산출물이 현재 저장본에는 없다.
- 따라서 이 출력만으로 "다음 연구 방향이 확정되었다"고 말할 수는 없다.

즉 6번은 계획 문서로는 성공했지만, 넓은 폭의 후속 검증 실험 자체를 대신하지는 못한다.

### 4.3 이번 결과가 시사하는 것

- 5번과 6번을 합쳐 보면, 지금 단계에서 바로 텍스트 독립변수나 데이터마트를 더 붙이는 것은 해석 리스크가 크다.
- 먼저 안정화 순서와 gate 기준을 고정하는 것이 필요하다는 점은 분명해졌다.
- 그러나 실제로 다음 방향을 고르려면, 더 넓은 모델군과 앙상블군을 포함하는 새 번호 실험이 필요하다.

## 5. 왜 추가 연구가 필요한가

5번은 대표 5개 구조를 약식으로 진단한 단계였다. 하지만 시계열 모델은 같은 계열이라도 변형 방식에 따라 학습 거동이 달라질 수 있다. 예를 들어 LSTM 기반이라도 residual, stacked, hybrid, auto-regressive head 조합에 따라 collapse 양상이 다를 수 있고, convolutional 계열도 TCN과 ModernTCN류가 다르게 반응할 수 있다.

또한 사용자는 단일 모델뿐 아니라 `Autoformer-like`, `PatchTST-like`, `DLinear/NLinear-like`, `TimesNet/TimeXer-like`, `iTransformer-like`, `ModernTCN-like`, `Mamba-like` 그리고 `LSTM + Autoformer`, `TCN + Transformer` 같은 혼합 앙상블까지 보고 전체 흐름을 판단하고자 한다.

이 범위는 6번 current output이 다루는 범위를 넘어선다. 따라서 6번을 억지로 확장하기보다, 기존 의미를 보존한 채 7번으로 새 실험을 분리하는 것이 맞다.

## 6. 다음 스텝

### 6.1 바로 다음 연구 파일

다음 단계는 6번을 덮어쓰는 것이 아니라 7번을 새 번호 실험으로 추가하는 것이다.

- `7번`: 넓은 폭의 후속 확장 실험 오케스트레이터
- 목적: 더 다양한 단일 모델군과 앙상블군을 포함해 전체 흐름을 한 번에 파악
- 역할: 6번의 gate 기준을 반영하되, 더 넓은 모델 변형과 앙상블을 체계적으로 비교

### 6.2 7번에서 반드시 포함할 것

- 기본 5개 구조 유지: `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`
- 확장 단일 모델군: `Autoformer-like`, `PatchTST-like`, `DLinear/NLinear-like`, `TimesNet/TimeXer-like`, `iTransformer-like`, `ModernTCN-like`, `Mamba-like`
- 앙상블군: `LSTM + Autoformer`, `TCN + Transformer`, `Linear + sequence model`, `validation weighted top-k ensemble`
- suite: `breadth_probe`, `ensemble_probe`, `normalization_cross_check`, `loss_cross_check`, `scale_confirmation`

### 6.3 보고서 작성 원칙

7번 보고서도 5번과 6번을 참조하더라도 핵심 설명을 생략하면 안 된다. 새 보고서는 그 문서만 읽어도 바로 이해할 수 있도록 다음 내용을 다시 적어야 한다.

- 각 알고리즘/앙상블이 무엇인지
- 왜 이번 단계에서 포함했는지
- loss/objective와 normalization이 무엇인지
- 성능 지표와 collapse 지표를 어떻게 읽는지
- 좋은 그림과 나쁜 그림이 각각 무엇인지
- 이번 결과가 무엇을 시사하는지

## 7. 관련 링크

- 5번 결과 보고서: `test/results/5_optimization_diagnostics_quick_probe_20260613.md`
- 6번 실행 계획서: `test/experiment_specs/6_optimization_stabilization_plan_20260613.md`
- 6번 오케스트레이터: `test/models/6_optimization_stabilization_test.ipynb`
- 6번 diff 추적 미러: `test/models/6_optimization_stabilization_test.py`

## 8. 결론

6번은 "실제 대규모 후속 실험 결과"가 아니라 "어떻게 후속 실험을 할지 고정한 오케스트레이터"라는 점에서 의미가 있다. 현재 저장된 결과는 그 의도에는 맞게 나왔지만, 사용자가 원하는 넓은 폭의 방향 판단 근거를 제공하기에는 부족하다. 따라서 6번은 중간 연결 문서로 보존하고, 다음 판단용 실험은 7번으로 분리하는 것이 현재 연구 흐름에 가장 맞다.
