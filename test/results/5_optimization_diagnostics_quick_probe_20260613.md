# 5. 최적화 학습 과정 진단 보고서

## 초록

- 문제: 비정상 업비트 15분봉에서는 손실함수 값이 빠르게 감소해도, 모델이 진짜 패턴을 학습했다기보다 `직전가를 거의 그대로 따르기`, `다음 수익률을 0 근처로 눌러버리기`, `변동성을 과도하게 축소한 평평한 예측` 같은 shortcut 해로 수렴할 수 있다.
- 방법: `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer` 5개 대표군으로 `level_mse`, `return_huber`, `directional_hybrid`를 비교하고, loss curve, gap, gradient, persistence gap, collapse score를 함께 봤다.
- 결과: 가장 덜 나쁜 후보는 `tcn_return_directional_hybrid`이며, test 기준 `persistence_gap = -10811.593750`, `sign agreement = 0.6234`로 유일하게 naive persistence를 소폭 하회했다. 다만 개선 폭이 매우 작고 variance ratio도 낮아, 바로 실전 채택을 말할 단계는 아니다.
- 의미: 이번 실험의 핵심은 "loss가 줄었는가"가 아니라 "그 감소가 실제 예측력인지, 아니면 비정상 시계열이 허용한 쉬운 해인지"를 5개 대표 아키텍처와 3개 목적함수에서 구조적으로 가려내는 데 있다.

### 초록 해석

- 훈련 손실이 빨리 내려갔다는 것은 "모델이 답을 찾기 시작했다"는 신호일 수는 있다. 하지만 금융 시계열에서는 그 답이 좋은 답인지, 너무 쉬운 답인지 따로 구분해야 한다.
- 그래서 이 보고서는 `훈련용 데이터에서 얼마나 잘 맞았는가`와 `처음 보는 구간에서도 비슷하게 맞는가`, `직전 값 복사보다 나은가`, `오르내림 방향은 맞히는가`, `예측이 너무 평평해지지 않았는가`를 따로 나눠 본다.
- 이번 결과는 일부 목적함수에서 "학습은 된다"는 사실을 보여줬지만, 그 학습이 곧바로 실전형 예측으로 이어진 것은 아니었다.
- 따라서 다음 단계는 변수를 더 붙이는 것보다, `무엇을 맞히게 할지`, `어떤 벌점을 줄지`, `쉬운 답으로 도망치면 어떻게 탈락시킬지`를 먼저 다듬는 쪽이 더 중요하다.

### 케이스 이름을 먼저 읽는 법

이 보고서에서 각 실험 케이스는 `모델구조_목적함수` 형식으로 이름을 붙였다. 예를 들어 `linear_level_mse`는 `Linear 구조`로 `가격 레벨`을 직접 예측하고, 그 오차를 `MSE`로 벌점 준 케이스라는 뜻이다.

- 앞부분은 `어떤 구조로 예측했는가`를 뜻한다.
  - `linear`: 가장 단순한 선형 기준선
  - `lstm`: 순환 메모리 구조
  - `gru`: 더 가벼운 순환 구조
  - `tcn`: 1D convolution 기반 구조
  - `transformer`: attention 기반 구조
- 뒷부분은 `무엇을 맞히게 했고, 어떤 벌점으로 학습시켰는가`를 뜻한다.
  - `level_mse`: 다음 가격 수준을 직접 맞히게 하고 큰 오차를 강하게 벌점
  - `return_huber`: 다음 수익률을 맞히게 하고 이상치에 덜 민감한 Huber 손실 사용
  - `directional_hybrid`: 수익률 크기뿐 아니라 방향을 틀리면 추가로 불리해지도록 설계한 혼합 목적함수

### 케이스를 해석하는 공통 틀

아래 개별 결과를 읽을 때는 항상 같은 순서로 본다.

1. 이 케이스가 무엇을 시험하려고 만든 조합인지 본다.
2. `훈련선`과 `검증선`이 같이 내려가는지 본다.
3. 두 선이 벌어지는지, 아니면 비슷하게 움직이는지 본다.
4. 단순 복사 기준선을 이겼는지 본다.
5. 방향을 맞히는지, 아니면 예측이 너무 평평해졌는지 본다.

즉, 케이스별 결과는 "숫자 공개"가 목적이 아니라, `이 조합이 어떤 종류의 실패를 보였는지`를 진단하는 데 목적이 있다.

### 교수님께 한 문장으로 말하면

- 이번 결과는 일부 objective가 학습 신호를 받긴 했지만, validation 일반화와 persistence baseline 돌파가 아직 충분하지 않아 최종 모델 선정보다는 objective/target 재조정이 먼저라는 뜻이다.

## 1. 서론

### 1.1 왜 이 실험이 필요한가

비정상 금융 시계열은 loss만 줄여서는 답이 안 나온다. 모델이 겉으로는 학습되는 것처럼 보여도, 실제로는 직전가 복사나 0 수익률 예측 같은 쉬운 해로 붕괴하기 쉽다.
그래서 이번 실험은 성능 순위를 뽑는 단계가 아니라, 어떤 objective가 잘못된 쉬운 해를 허용하는지 먼저 확인하는 진단 단계다.

### 1.1.1 이전 단계에서 이미 관찰된 경고 신호

아래 그림은 이전 고급 시계열 실험에서 추출한 학습 손실 곡선이다. 로컬 노트북 출력에서 파싱한 이미지이며, 이번 보고서의 문제 정의를 뒷받침하는 근거로 사용한다.

![이전 단계 학습 손실 곡선](../images/2_time_series_advance_test_plot_6.png)

- 근거 이미지: `test/images/2_time_series_advance_test_plot_6.png`
- 그림 해석 1: Transformer 계열과 RNN/CNN 계열 대부분이 초반 1~2 epoch 안에 손실을 거의 0 근처까지 급격히 낮춘다.
- 그림 해석 2: 이렇게 너무 빠른 수렴이 모든 모델에서 비슷하게 나타나면, "모델이 시장 구조를 잘 배웠다"기보다 "현재 목적함수가 허용한 쉬운 답을 재빨리 찾았다"는 해석이 더 타당하다.
- 그림 해석 3: 만약 기울기 소실이 주된 문제였다면 학습이 잘 안 되거나 손실이 평평하게 남는 모습이 중심이 되어야 한다. 그런데 여기서는 오히려 손실이 너무 쉽게 내려간다.
- 따라서 이전 단계의 핵심 문제는 `gradient vanishing`보다 `비정상성과 높은 자기상관이 결합된 환경에서 목적함수가 직전가 복사/무변화 예측/축소된 분산 예측을 지나치게 쉽게 허용하는 것`에 가깝다.

### 1.2 이 실험이 풀려는 문제

현재 연구는 독립변수 확장과 예측 성능 개선이 목적이지만, objective가 잘못되면 변수만 늘려도 모델은 계속 무너진다. 그래서 먼저 loss, head, architecture가 안전한지 봐야 한다.
즉, 이 실험은 '무슨 모델이 최고냐'보다 '어디서 잘못 무너지는지'를 찾기 위한 것이다.

### 1.3 연구 질문

- raw next-close 회귀가 copy-risk를 키우는가
- return target + Huber가 평평한 해를 덜 허용하는가
- directional hybrid가 0 수익률 붕괴를 줄이는가
- 같은 objective에서도 Linear, LSTM, GRU, TCN, Transformer의 붕괴 양상이 다른가

### 1.4 해석 기준

- 왼쪽 위 그래프에서 `train` 선과 `validation` 선이 함께 내려가면, 훈련 구간과 검증 구간 모두에서 손실이 줄고 있다는 뜻이다.
- 반대로 `train` 선만 더 많이 내려가고 `validation` 선은 위쪽에 남아 있으면, 모델이 훈련 구간에는 익숙해졌지만 새 구간에는 약하다는 뜻이다.
- 오른쪽 아래 그래프의 가로 기준선은 "직전 값을 그대로 내놓는 아주 단순한 기준선"이다. 모델 선이 그 기준선 위에 있으면 단순 복사보다 못한 것이고, 아래로 내려오면 그제서야 기준선을 이긴 것이다.
- `sign agreement`가 `0.5` 안팎이라는 말은, 오를지 내릴지를 거의 반반 수준으로만 맞힌다는 뜻이다. 쉽게 말해 동전 던지기와 크게 다르지 않다.
- `collapse score`는 참고용 종합 경고등이다. 이 값 하나만 보고 좋다 나쁘다를 결정하면 안 된다.
- 특히 이번 실험에서는 "손실이 빨리 줄었다"는 사실만으로 기울기 문제가 해결됐다고 보지 않는다. 오히려 목적함수가 너무 쉬운 답을 허용해서 모델이 빨리 멈춘 것일 수 있다.

## 2. 방법론

### 2.1 실행 설정

- Suite: `quick_probe`
- Feature set: `optimization_probe`
- Feature count: `6`
- Sequence length: `32`
- Epochs: `30`
- Max rows: `35040`
- Max windows: `1024`
- Window stride: `4`
- Result directory: `test/results`
- Image directory: `test/images`
- Save artifacts: `False`
- Save CSV: `False`

### 2.1.1 `optimization_probe` feature set은 무엇인가

이번 실험의 feature set 이름은 `optimization_probe`이지만, 이름 자체보다 중요한 것은 "왜 이 6개 변수를 골랐는가"이다. 이번 실험은 최종 예측 성능 대결이 아니라 `학습이 왜 무너지는지`를 보는 진단 실험이므로, 너무 많은 변수를 넣기보다 역할이 분명한 소수 변수로 구성했다.

- `log_return_1`
  - 의미: 직전 1개 구간의 로그수익률
  - 왜 넣었나: 가격 수준 자체보다 더 정상성에 가까운 짧은 변화율 신호를 주기 위해서다.
- `return_4`
  - 의미: 최근 4개 구간 누적 수익률
  - 왜 넣었나: 아주 짧은 잡음만 보지 않고, 조금 더 넓은 짧은 추세를 보게 하기 위해서다.
- `realized_vol_16`
  - 의미: 최근 16개 구간의 실현 변동성
  - 왜 넣었나: 지금이 조용한 구간인지, 흔들리는 구간인지 모델이 구분하게 하려는 목적이다.
- `hl_range_pct`
  - 의미: 고가-저가 범위를 가격 대비 비율로 바꾼 값
  - 왜 넣었나: 종가만으로 안 보이는 봉 내부의 흔들림 크기를 반영하기 위해서다.
- `volume_z_96`
  - 의미: 거래량을 최근 96개 구간 기준으로 표준화한 값
  - 왜 넣었나: 지금 거래량이 평소보다 과열인지 위축인지 상대적으로 판단하게 하려는 목적이다.
- `spread_proxy`
  - 의미: 거래 마찰이나 체결 난도를 간접적으로 반영하는 프록시 변수
  - 왜 넣었나: 같은 가격 변화라도 유동성 상태가 다르면 학습 난이도와 예측 안정성이 달라질 수 있기 때문이다.

### 2.1.2 왜 하필 이 6개만 썼는가

- 첫째, 이번 실험 목적은 `최적의 alpha 찾기`가 아니라 `목적함수와 구조가 쉬운 해로 무너지는지 점검`하는 것이다.
- 둘째, 변수를 너무 많이 넣으면 "입력이 많아서 얼핏 그럴듯해 보이는 결과"와 "목적함수 자체가 괜찮아서 버틴 결과"를 구분하기 어려워진다.
- 셋째, 이 6개는 각각 `단기 변화`, `짧은 추세`, `변동성`, `봉 내부 진폭`, `상대 거래량`, `유동성/마찰`을 대표하도록 골랐기 때문에, 작은 세트지만 시장 상태의 핵심 축을 빠르게 훑는 데 적합하다.
- 따라서 `optimization_probe`는 성능 극대화용 feature set이 아니라, `학습 붕괴 진단용 최소 유효 세트`라고 이해하는 편이 맞다.

### 2.2 데이터 및 처리 조건

## 3. 데이터 및 처리 조건

아래 조건은 이번 최적화 진단이 어떤 데이터에서 수행되었는지 설명한다. 손실함수만 보는 것이 아니라, 데이터 기간과 분할 구조를 함께 봐야 학습 곡선의 의미를 해석할 수 있다.

- DuckDB 경로: `data/upbit_data.db`
- 가격 테이블: `btc_15m_advance`
- 티커 필터: `ALL`
- 사용 행 수: `17,181`
- 종목 수: `1`
- 기간: `2025-12-15 15:00:00` ~ `2026-06-12 23:45:00`
- 입력 변수 수: `6`
- 입력 변수: `log_return_1, return_4, realized_vol_16, hl_range_pct, volume_z_96, spread_proxy`
- 시퀀스 분할: train `716`, validation `154`, test `154`

### 3.0.1 변수 구성이 이번 문제와 맞는 이유

이번 문제는 "비정상 가격 수준을 그대로 맞히는 과정에서 모델이 쉬운 답으로 무너지느냐"를 보는 것이다. 그래서 입력 변수도 가격 레벨 자체보다 `변화율`, `변동성`, `상대적 거래 상태`를 중심으로 잡았다.

- 수익률 계열 변수(`log_return_1`, `return_4`)는 가격 수준 복사 유혹을 줄이고, 변화 자체를 보게 만든다.
- 변동성/범위 변수(`realized_vol_16`, `hl_range_pct`)는 지금 시장이 조용한지 거친지 알려줘서, 모델이 모든 구간을 한 가지 방식으로 보지 않게 돕는다.
- 상태 변수(`volume_z_96`, `spread_proxy`)는 거래량 과열이나 유동성 악화처럼, 같은 가격 변화라도 다른 해석이 필요한 상황을 보완한다.

즉 이번 feature set은 "많이 넣어서 맞히기"보다 "`쉬운 해 붕괴`를 덜 유도하는 정보만 먼저 넣어본 구성"이라고 이해하면 된다.

### 3.1 기초 통계량

| 항목 | 값 | 해석 |
| --- | ---: | --- |
| Close 평균 | 113,379,749.8399 | 가격 레벨의 중심 크기 |
| Close 표준편차 | 13,110,598.7898 | 가격 레벨의 변동 폭 |
| Close 최소값 | 89,250,000.0000 | 분석 구간 내 최저 가격 |
| Close 중앙값 | 111,984,000.0000 | 극단값 영향을 줄인 중심 가격 |
| Close 최대값 | 142,968,000.0000 | 분석 구간 내 최고 가격 |
| 다음 로그수익률 평균 | -0.00001871 | 다음 15분 변동의 평균 방향 |
| 다음 로그수익률 표준편차 | 0.00239308 | 다음 15분 변동성 크기 |
| 절대 로그수익률 평균 | 0.00155258 | 평균적인 1-step 변동 강도 |
| 상승 비율 | 0.4960 | 다음 봉이 상승한 비율, 0.5 근처면 방향 예측 난도 높음 |

### 2.3 방법론 및 진단 지표 정의

이 절은 결과를 보기 전에 각 지표가 무엇을 의미하는지 정의한다. 본 실험의 목적은 최종 수익률을 주장하는 것이 아니라, 손실함수와 모델 구조가 비정상 금융 시계열에서 어떤 방식으로 최적화되는지 진단하는 것이다.

### 2.3.0 아주 짧은 용어 번역

- `train`: 모델이 직접 보고 배우는 훈련 구간
- `validation`: 학습 중간에 "처음 보는 문제"처럼 따로 검사하는 검증 구간
- `test`: 마지막에 최종 확인용으로 남겨둔 구간
- `loss`: 모델이 얼마나 틀렸는지를 숫자로 벌점화한 값
- `gap`: 두 값 사이의 차이
- `gradient`: 모델이 다음 번에 얼마나, 어느 방향으로 수정되어야 하는지 알려주는 학습 신호
- `baseline`: "이 정도는 최소한 이겨야 한다"는 비교 기준
- `persistence`: 다음 값이 지금 값과 비슷할 것이라고 보는 가장 단순한 예측
- `collapse`: 모델이 진짜 예측 대신 너무 쉬운 답으로 무너지는 현상

### 2.3.1 Train/Validation Loss Curve

- 개념: epoch가 지날수록 `훈련용 데이터에서의 오차`와 `처음 보는 검증 데이터에서의 오차`가 어떻게 변하는지 보여주는 가장 기본적인 그래프다.
- 사용 이유: 모델이 단순히 훈련 데이터를 외우는지, 아니면 처음 보는 구간에서도 비슷하게 작동하는지 구분하기 위해 사용한다.
- 정의: `train_loss = L(y_train, yhat_train)`, `validation_loss = L(y_val, yhat_val)`이며, 본 실험에서는 서로 다른 손실함수끼리 비교하기 쉽게 첫 epoch 값을 1.0으로 맞춘 상대 지표도 함께 본다.
- 해석 예시: 두 선이 같이 내려가면 "훈련 구간에서도 좋아지고, 검증 구간에서도 같이 좋아진다"는 뜻이다. 반대로 훈련선만 빨리 내려가고 검증선이 덜 움직이면, 겉보기 학습은 됐지만 실제 예측력은 약할 수 있다.
- 장점: 가장 직관적으로 최적화 과정의 수렴, 정체, 발산을 확인할 수 있다.
- 한계: 금융 시계열에서는 validation loss가 낮아도 방향성 또는 매매 가능성이 보장되지 않는다. 따라서 persistence gap, sign agreement 같은 보조 지표가 필요하다.

### 2.3.2 Generalization Gap

- 개념: 같은 시점에서 `검증선이 훈련선보다 얼마나 위에 떠 있는지`를 숫자로 적어놓은 것이다.
- 사용 이유: 훈련 구간에서는 잘 맞는데, 미래 구간으로 가면 성능이 떨어지는 상황을 눈으로 더 쉽게 잡기 위해 사용한다.
- 정의: `gap = validation_loss_index - train_loss_index`.
- 해석 예시: 이 값이 커진다는 것은 검증선이 훈련선보다 더 위로 벌어진다는 뜻이다. 즉, 훈련 구간에서는 잘 맞아도 새 구간에서는 덜 맞는다는 의미다.
- 장점: 과적합 여부를 빠르게 볼 수 있다.
- 한계: train/validation 구간 자체가 regime shift를 포함하면 gap은 과적합뿐 아니라 시장 국면 변화의 신호일 수도 있다.

### 2.3.3 Gradient Norm Before Clipping

- 개념: 가중치를 얼마나 세게 업데이트하려는지 보여주는 학습 신호의 크기다.
- 사용 이유: 손실함수가 너무 거칠어서 업데이트가 과하게 튀는지, 반대로 너무 작은 신호만 남아서 학습이 멈춰 가는지 확인하기 위해 사용한다.
- 정의: `||grad|| = sqrt(sum_i grad_i^2)`.
- 해석 예시: 값이 계속 너무 크면 모델이 매번 과격하게 움직이고 있다는 뜻일 수 있다. 반대로 값이 너무 빨리 작아지면 "이제 배울 게 거의 없다"는 신호일 수도 있고, "쉬운 답을 이미 찾아서 더는 움직이지 않는다"는 뜻일 수도 있다.
- 장점: loss curve만으로 보이지 않는 최적화 안정성을 확인할 수 있다.
- 한계: gradient norm이 안정적이라고 해서 예측력이 좋다는 뜻은 아니다. 안정적으로 잘못된 해에 수렴할 수도 있다.

### 2.3.4 Persistence Gap

- 개념: 모델이 `직전 값을 그대로 내놓는 아주 단순한 기준선`보다 얼마나 잘했는지, 혹은 못했는지를 보여주는 비교 점수다.
- 사용 이유: 금융 가격 수준 데이터는 원래 이전 값과 매우 비슷한 경우가 많아서, 복잡한 딥러닝 모델이 오히려 단순 복사보다 못한 상황이 자주 생기기 때문이다.
- 정의: `persistence_gap = MAE(model_prediction, y_next) - MAE(current_price, y_next)`.
- 해석 예시: 이 값의 기준점은 `0`이다. `0`은 "모델과 단순 복사가 똑같다"는 뜻이다. 숫자가 `0`보다 크면 모델이 단순 복사보다 더 틀렸다는 뜻이고, `0`보다 작으면 그제서야 단순 복사보다 조금이라도 낫다는 뜻이다.
- 장점: 가격 레벨 예측에서 발생하는 기만적 RMSE 착시를 방지한다.
- 한계: persistence baseline은 가격 레벨 기준이다. 방향성 전략이나 변동성 예측에서는 별도의 경제적 baseline도 함께 봐야 한다.

### 2.3.5 Collapse Score

- 개념: 모델이 예측 변동성을 잃거나, 0 수익률만 내거나, 직전 가격과 과도하게 붙는 쉬운 해로 도망가는 위험을 합친 보조 점수다.
- 사용 이유: 손실함수 최적화가 허용하는 가장 쉬운 해가 실제 예측이 아니라 평균/무변화/복사일 수 있기 때문이다.
- 정의: 본 코드에서는 낮은 prediction variance, near-zero return share, copy alignment를 가중 결합한다.
- 해석 예시: 이 점수는 자동차 계기판의 경고등에 가깝다. 낮으면 일단 덜 위험하다고 볼 수 있지만, 이것만 낮다고 좋은 모델이라고 결론내리면 안 된다. 다른 그래프에서 단순 복사보다 못하면 여전히 탈락이다.
- 장점: 모델이 명백히 평평한 예측 또는 복사 예측으로 무너지는지 빠르게 확인할 수 있다.
- 한계: 단일 성능 지표가 아니다. collapse score만 낮다고 좋은 모델이 아니다.

### 2.3.6 Near-Zero Return Share

- 개념: 모델이 예측한 다음 수익률이 거의 0에 가까운 비율이다.
- 사용 이유: 수익률 예측 모델이 '다음에도 거의 안 움직인다'는 평균 해로 붕괴하는지 확인하기 위해 사용한다.
- 정의: `mean(abs(predicted_return) < 1e-4)`.
- 해석 예시: 이 값이 높다는 것은 모델이 자꾸 "다음에도 거의 그대로일 것"이라고 답하고 있다는 뜻이다. 실제로 시장이 조용한 구간일 수도 있지만, 너무 높으면 모델이 안전한 쉬운 답으로 숨고 있을 가능성이 있다.
- 장점: 수익률 target에서 평균 수렴/무변화 예측을 탐지하기 쉽다.
- 한계: 실제 시장이 저변동 구간이면 near-zero 비율이 높게 나올 수 있으므로, realized volatility와 함께 해석해야 한다.

### 2.3.7 Sign Agreement

- 개념: 예측한 가격 변화 방향과 실제 가격 변화 방향의 부호가 일치한 비율이다.
- 사용 이유: 가격 오차가 작아도 상승/하락 방향을 못 맞히면 트레이딩 관점에서 유효성이 낮기 때문이다.
- 정의: `mean(sign(predicted_delta) == sign(actual_delta))`.
- 해석 예시: `0.5` 근처면 오를지 내릴지를 거의 반반 수준으로만 맞힌다는 뜻이다. 숫자가 조금 높아 보이더라도, 실제 매매에서는 수수료와 슬리피지를 넘길 만큼의 차이인지 더 따져봐야 한다.
- 장점: 가격 레벨 RMSE의 착시를 보완한다.
- 한계: 방향만 맞고 크기를 틀리면 실제 손익은 나쁠 수 있다.

### 2.3.8 알고리즘 가족 설명

- `Linear`: 입력 윈도우를 한 번에 펴서 가장 단순한 선형 조합으로 예측한다. 장점은 빠르고 해석이 쉬우며, 단점은 시계열 순서 정보를 거의 직접 쓰지 못한다는 점이다.
- `LSTM`: 과거 상태를 게이트로 누적하는 순환 구조다. 장점은 시계열의 순서를 기억할 수 있다는 점이고, 단점은 데이터가 적거나 objective가 거칠면 쉽게 평평한 해로 수렴할 수 있다는 점이다.
- `GRU`: LSTM보다 게이트가 단순한 순환 구조다. 장점은 더 가볍고 빠르게 학습된다는 점이며, 단점은 경우에 따라 복잡한 장기 의존성을 덜 잡을 수 있다는 점이다.
- `TCN`: dilation이 있는 1D convolution으로 국소 패턴과 다중 길이 receptive field를 동시에 보는 구조다. 순환 없이도 시계열 구조를 잡을 수 있어, shortcut 붕괴를 덜 허용하는지 확인하기 좋은 비교군이다.
- `Transformer`: attention으로 여러 시점 간 상호작용을 직접 본다. 표현력은 강하지만, 데이터 규모가 작거나 목적함수가 허술하면 오히려 shortcut 해를 더 빠르게 찾을 수도 있다.
- 이번 실험에서 5개 아키텍처를 함께 두는 이유는, 복잡도가 높을수록 항상 좋은 것이 아니라는 점을 확인하기 위해서다. 같은 objective에서도 아키텍처가 다르면 붕괴 방식이 달라질 수 있다.

## 3. 결과

### 3.1 전체 그림 먼저 읽기

아래 그림은 전체 케이스를 한 번에 비교하기 위한 요약 그림이다. 이 그림만 보면 '누가 loss를 줄였나'는 보이지만, 아직 '왜 그게 좋은가'는 안 보인다. 그래서 학습 곡선 뒤에 일반화 갭, baseline 비교, 붕괴 지표를 같이 붙였다.

노트북 출력 셀에 표시되는 `training_figure`를 먼저 확인한다.

### 3.1.1 이번 결과를 한 문장으로 먼저 요약하면

- `level_mse` 계열은 가격 레벨 자기상관이 너무 강해서 5개 대표군 전체가 사실상 복사형 문제 설정에 끌려갔다.
- `return_huber` 계열은 손실은 잘 내려갔지만, 여러 케이스에서 0 수익률 근처 또는 축소된 분산 예측으로 수렴하는 경향이 남았다.
- `directional_hybrid` 계열은 방향성 제약 덕분에 shortcut 억제 시도는 보였지만, 그 효과가 모든 구조에서 안정적으로 재현되지는 않았다.
- 5개 대표군 중에서는 `TCN`이 가장 덜 무너졌고, `Transformer`는 표현력 대비 baseline 방어가 가장 취약한 쪽에 속했다.

### 3.2 그래프별 축과 해석 기준

- `Train/Validation Objective Loss Index`: x축은 epoch, y축은 시작 시점 대비 손실이 얼마나 줄었는지다. 왼쪽 위 패널이다. 여기서는 `train` 선과 `validation` 선이 같이 내려가는지, 혹은 한쪽만 내려가는지를 먼저 본다.
- `Validation - Train Gap`: x축은 epoch, y축은 `검증 손실 - 훈련 손실`이다. 오른쪽 위 패널이다. 이 그래프의 핵심은 숫자 부호보다도 "검증선이 훈련선보다 얼마나 더 위에 남아 있는가"다. 선이 위로 올라가면 두 성능 차이가 커진다는 뜻이다.
- `Gradient Norm Before Clipping`: x축은 epoch, y축은 가중치 업데이트 신호의 크기다. 왼쪽 아래 패널이다. 값이 너무 크면 학습이 거칠고, 너무 빨리 작아지면 학습이 일찍 멈추었을 가능성이 있다.
- `Validation Persistence Gap`: x축은 epoch, y축은 `모델 오차 - 직전값 복사 오차`다. 오른쪽 아래 패널이다. 가운데 기준선 `0`은 "모델과 단순 복사가 똑같다"는 뜻이다. 선이 그 위에 있으면 단순 복사보다 못하고, 아래로 내려오면 단순 복사보다 낫다.
- 요약하면, 왼쪽 위는 "학습이 되느냐", 오른쪽 위는 "훈련 때만 잘하느냐", 왼쪽 아래는 "업데이트가 안정적이냐", 오른쪽 아래는 "단순 복사를 이기느냐"를 보여준다.

### 3.3 현재 결과를 읽는 핵심 기준

- 훈련선만 내려가고 검증선은 위쪽에 남아 있으면, 모델은 훈련 데이터에는 적응했지만 새로운 구간에서는 그만큼 통하지 않는다는 뜻이다.
- 검증선이 거의 움직이지 않으면, 지금의 입력과 손실함수 조합이 가격 변화를 배울 만한 신호를 충분히 주지 못하고 있다는 뜻이다.
- persistence 그래프에서 선이 계속 기준선 위에 머물면, 아무리 복잡한 모델이라도 "직전 값을 그대로 내놓기"보다 못한 것이다.
- 그래서 이 실험에서는 단순히 `loss가 낮다`보다 `검증선도 같이 내려오는가`, `훈련선과 검증선의 간격이 줄어드는가`, `단순 복사 기준선 아래로 내려오는가`를 더 중요하게 본다.

### 3.4 그래프가 애매해 보일 때 해석

- 손실은 내려가는데 다른 그래프가 나빠지면, 모델이 "문제를 잘 푼 것"이 아니라 "점수만 잘 깎는 법을 찾은 것"일 수 있다.
- collapse score나 near-zero 비율이 겉보기에 나쁘지 않아도, persistence 그래프에서 기준선 위에 머물면 실제로는 단순 복사보다 못한 상태다.
- sign agreement가 `0.5` 부근이면, 오를지 내릴지조차 뚜렷하게 못 맞히는 상태라서 방향 전략으로도 조심해야 한다.
- gradient가 들쑥날쑥하면 학습이 거칠다는 뜻일 수 있고, 너무 빨리 작아지면 좋은 수렴이 아니라 쉬운 답에 빨리 멈춘 것일 수도 있다.
- 따라서 그래프는 항상 한 장씩 따로 보지 말고, `손실`, `훈련-검증 간격`, `업데이트 신호`, `단순 복사 대비 우위`를 같이 읽어야 한다.

### 3.4.1 대표군별 종합 해석

- `Linear`: 통제군 역할은 분명했다. 복잡한 구조 없이도 return 계열 loss는 줄일 수 있었지만 baseline을 못 넘는 경우가 많아, 문제의 핵심이 "모델이 너무 단순해서"만은 아니라는 점을 보여준다.
- `LSTM`: 시계열 순서를 활용하려 했지만 level_mse에서는 거의 개선이 없고, return 계열에서도 방향성과 baseline 돌파가 약했다. 즉 메모리 구조 자체가 이번 목적함수의 결함을 해결해주지는 못했다.
- `GRU`: LSTM보다 가벼워서 수렴은 잘 됐지만, 더 좋은 일반화로 이어지지는 않았다. 학습이 쉬워진 것과 유효한 예측이 된 것은 별개라는 점을 보여준다.
- `TCN`: 세 family 중 가장 실용적인 균형에 가까웠다. 특히 `directional_hybrid`에서 유일하게 persistence gap을 0 아래로 내렸다는 점이 중요하다. 다만 variance ratio가 낮아 예측이 아직 보수적으로 수축되어 있다는 한계가 남는다.
- `Transformer`: loss 감소 자체는 보이지만 baseline 방어가 매우 약했다. 이는 attention 구조가 나빠서라기보다, 현재 데이터 크기와 목적함수 조합에서는 표현력이 shortcut을 더 효율적으로 찾는 쪽으로 작동했을 가능성을 시사한다.

### 3.4.2 케이스별 짧은 해석

이 절은 개별 케이스를 처음 훑어볼 때 쓰는 요약표다. 이미 앞에서 `케이스 이름의 뜻`과 `읽는 순서`를 설명했기 때문에, 여기서는 "이 조합이 무엇을 시험했고 결과가 어땠는가"만 짧게 정리한다.

| 케이스 | 무엇을 시험한 조합인가 | 한 줄 해석 |
| --- | --- | --- |
| `linear_level_mse` | 가장 단순한 구조로 가격 수준 직접 회귀 | 학습 신호가 거의 살아나지 못했고, 단순 복사보다도 한참 약했다. |
| `linear_return_huber` | 선형 구조에서 수익률 예측 + 완만한 손실 | 손실은 줄었지만 새 구간 적응력과 baseline 돌파는 부족했다. |
| `linear_return_directional_hybrid` | 선형 구조에서 수익률 + 방향 벌점 | 방향 제약을 줘도 일반화가 약했고 단순 복사를 넘지 못했다. |
| `lstm_level_mse` | 순환 메모리 구조로 가격 수준 직접 회귀 | 메모리 구조를 써도 가격 수준 직접 회귀의 복사 위험을 해결하지 못했다. |
| `lstm_return_huber` | LSTM으로 수익률 예측 + Huber | 학습은 진행됐지만 예측력이 뚜렷하게 살아났다고 보긴 어려웠다. |
| `lstm_return_directional_hybrid` | LSTM으로 수익률 + 방향 벌점 | 방향 정보까지 넣었지만 검증 우위가 약했고 baseline도 넘지 못했다. |
| `gru_level_mse` | 더 가벼운 순환 구조로 가격 수준 직접 회귀 | 가벼워도 본질은 같아서, 가격 수준 직접 회귀의 한계를 그대로 드러냈다. |
| `gru_return_huber` | GRU로 수익률 예측 + Huber | 수렴은 비교적 쉬웠지만 그것이 좋은 예측으로 이어지지는 않았다. |
| `gru_return_directional_hybrid` | GRU로 수익률 + 방향 벌점 | 구조는 가벼웠지만 방향 제약의 이점이 안정적으로 남지 않았다. |
| `tcn_level_mse` | convolution 구조로 가격 수준 직접 회귀 | 구조를 바꿔도 가격 수준 직접 회귀는 여전히 복사형 함정에 머물렀다. |
| `tcn_return_huber` | TCN으로 수익률 예측 + Huber | 손실은 잘 줄었지만 예측이 지나치게 보수적으로 수축되는 경향이 있었다. |
| `tcn_return_directional_hybrid` | TCN으로 수익률 + 방향 벌점 | 이번 15개 중 가장 덜 무너졌고, test 기준으로만 보면 단순 복사를 소폭 이겼다. |
| `transformer_level_mse` | attention 구조로 가격 수준 직접 회귀 | 표현력이 강해도 잘못된 목적을 주면 좋은 답 대신 쉬운 답을 찾을 수 있음을 보여줬다. |
| `transformer_return_huber` | Transformer로 수익률 예측 + Huber | 학습은 됐지만 baseline 방어가 약해 실전형 예측이라고 보긴 어렵다. |
| `transformer_return_directional_hybrid` | Transformer로 수익률 + 방향 벌점 | 방향 벌점을 줘도 이번 데이터 규모에서는 shortcut 억제가 충분하지 않았다. |

### 3.5 모델별 학습 곡선

각 모델/손실함수 조합을 분리해서 본 그림이다. 여기서는 최적화 과정만 빠르게 확인하기 위해 케이스별로 분리했다. 중요한 건 '값이 있나'가 아니라 '그 값이 무슨 뜻인가'다.
아래 각 패널은 모두 같은 읽는 법을 따른다. 왼쪽 위에서는 `훈련선`과 `검증선`이 같이 내려가는지 본다. 오른쪽 위에서는 두 선이 서로 멀어지는지 본다. 왼쪽 아래에서는 학습 신호가 너무 거칠거나 너무 빨리 죽는지 본다. 오른쪽 아래에서는 모델 선이 `단순 복사 기준선` 아래로 내려오는지 본다.

개별 케이스에서 자주 나오는 표현도 먼저 정리하면 다음과 같다.

- `train이 내려가고 validation이 덜 내려간다`: 훈련 때는 맞지만 새 구간에서는 덜 맞는다.
- `gap이 커진다`: 훈련선과 검증선 사이가 더 벌어진다.
- `persistence gap이 기준선 위에 있다`: 단순 복사보다 못하다.
- `persistence gap이 기준선 아래로 내려온다`: 단순 복사보다 낫다.
- `sign agreement가 0.5 부근이다`: 오를지 내릴지를 반반 수준으로만 맞힌다.

#### 3.5.1 tcn_return_directional_hybrid
- 무엇을 시험했나: TCN이 짧은 국면 패턴을 convolution으로 잡을 때, 방향 벌점이 0 수익률 shortcut을 얼마나 줄이는지 확인한 케이스다.
- 좋은 그림이면: 훈련선과 검증선이 함께 내려가고, persistence gap 선이 단순 복사 기준선 아래로 안정적으로 내려가야 한다.
- 이번 그림은 왜 그렇지 않은가: test 기준으로는 단순 복사를 소폭 이겼지만, zero share가 높고 variance ratio가 낮아 예측 폭이 아직 보수적으로 눌려 있다.
- 결과 해석: 이번 15개 중 가장 덜 무너진 조합이다. `persistence_gap = -10811.593750`, `sign agreement = 0.6234`라서 후속 실험의 1순위 후보지만, 개선 폭이 작아 최종 모델이 아니라 다음 단계 검증 출발점으로 본다.

#### 3.5.2 lstm_return_directional_hybrid
- 무엇을 시험했나: LSTM의 순환 메모리가 방향 벌점과 결합될 때, 수익률 방향을 더 잘 붙잡는지 확인했다.
- 좋은 그림이면: 검증선이 훈련선과 같이 내려가고, 방향 벌점 덕분에 sign agreement가 0.5를 분명히 넘어야 한다.
- 이번 그림은 왜 그렇지 않은가: persistence gap이 기준선 위에 남아 단순 복사를 넘지 못했고, sign agreement도 `0.4740`으로 반반보다 낮았다.
- 결과 해석: 방향 벌점을 넣었지만 LSTM 구조가 이번 입력과 목적함수에서는 방향 신호를 안정적으로 살리지 못했다. 메모리 구조 자체가 shortcut 문제를 해결해주지는 않았다.

#### 3.5.3 lstm_return_huber
- 무엇을 시험했나: LSTM에 Huber 손실을 붙여 이상치에 덜 흔들리는 수익률 학습이 되는지 확인했다.
- 좋은 그림이면: 손실선이 부드럽게 내려가면서도 persistence gap이 기준선 아래로 이동해야 한다.
- 이번 그림은 왜 그렇지 않은가: 최적화는 움직였지만 test 기준 `persistence_gap = 95688.156250`으로 단순 복사보다 못했고, 방향성도 `0.4610`에 그쳤다.
- 결과 해석: Huber가 학습을 부드럽게 만들 수는 있지만, 그 자체로 예측 가능한 신호를 만들어주지는 못했다. 이 케이스는 "잘 내려가는 손실"과 "쓸 수 있는 예측"이 다르다는 예시다.

#### 3.5.4 gru_return_huber
- 무엇을 시험했나: 더 가벼운 순환 구조인 GRU가 Huber 손실과 결합될 때 LSTM보다 안정적인지 확인했다.
- 좋은 그림이면: LSTM보다 검증선이 덜 흔들리고 persistence gap이 더 빠르게 기준선 아래로 내려와야 한다.
- 이번 그림은 왜 그렇지 않은가: sign agreement는 `0.5455`로 약간 높지만, `persistence_gap = 154568.875000`이라 가격 오차 기준에서는 단순 복사를 넘지 못했다.
- 결과 해석: 방향을 조금 맞히는 듯해도 가격 오차 기준 방어선은 실패했다. GRU의 가벼운 수렴이 좋은 일반화로 이어졌다고 보기 어렵다.

#### 3.5.5 gru_return_directional_hybrid
- 무엇을 시험했나: GRU에 방향 벌점을 더하면 Huber보다 shortcut 억제가 좋아지는지 확인했다.
- 좋은 그림이면: 방향 벌점이 들어간 만큼 sign agreement가 올라가고, zero-return 쪽으로 눌리는 현상이 줄어야 한다.
- 이번 그림은 왜 그렇지 않은가: sign agreement가 `0.4351`로 낮아 방향 벌점의 효과가 남지 않았고, persistence gap도 기준선 위에 머물렀다.
- 결과 해석: GRU에서는 directional hybrid가 오히려 안정적인 방향 신호로 이어지지 못했다. 이 조합은 다음 단계 우선순위에서 낮게 둔다.

#### 3.5.6 linear_return_huber
- 무엇을 시험했나: 복잡한 구조 없이도 수익률 Huber만으로 최소한의 예측 신호가 나오는지 확인하는 통제군이다.
- 좋은 그림이면: 단순한 모델이라도 검증선이 함께 내려가고 persistence gap이 기준선 근처까지 접근해야 한다.
- 이번 그림은 왜 그렇지 않은가: 손실은 움직였지만 `persistence_gap = 208957.187500`, `sign agreement = 0.3961`로 baseline과 방향성 모두 약했다.
- 결과 해석: 문제의 핵심이 단지 모델 복잡도 부족은 아니라는 점을 보여준다. 목적함수와 target 설계가 먼저 안전해야 한다.

#### 3.5.7 tcn_return_huber
- 무엇을 시험했나: TCN의 convolutional smoothing이 Huber 손실과 결합될 때 안정적인 수익률 예측으로 이어지는지 확인했다.
- 좋은 그림이면: 손실선이 내려가면서도 예측 분산이 지나치게 줄지 않고, persistence gap이 기준선 아래로 내려와야 한다.
- 이번 그림은 왜 그렇지 않은가: `persistence_gap = 23272.093750`으로 기준선에 가까워졌지만 넘지는 못했고, sign agreement도 `0.4091`로 낮았다.
- 결과 해석: TCN 구조는 다른 구조보다 덜 무너지는 편이지만, Huber만으로는 무변화 예측 성향을 충분히 밀어내지 못했다.

#### 3.5.8 linear_return_directional_hybrid
- 무엇을 시험했나: 선형 통제군에서도 방향 벌점이 방향성 개선을 만들 수 있는지 확인했다.
- 좋은 그림이면: 단순 구조라도 sign agreement가 올라가고, persistence gap이 Huber보다 개선되어야 한다.
- 이번 그림은 왜 그렇지 않은가: sign agreement는 `0.5519`로 약간 높지만 `persistence_gap = 857286.843750`으로 가격 오차는 크게 나빠졌다.
- 결과 해석: 방향 벌점을 넣으면 방향 비율만 살짝 좋아질 수 있지만, 가격 오차와 baseline 방어가 함께 무너지면 실전 후보가 될 수 없다.

#### 3.5.9 transformer_return_huber
- 무엇을 시험했나: 표현력이 큰 Transformer가 Huber 손실만으로 더 풍부한 수익률 패턴을 잡는지 확인했다.
- 좋은 그림이면: 복잡한 구조답게 검증선과 persistence gap이 동시에 개선되어야 하며, 과도한 분산 확대가 없어야 한다.
- 이번 그림은 왜 그렇지 않은가: sign agreement는 `0.5844`로 좋아 보이지만 `persistence_gap = 997905.718750`으로 baseline 방어가 크게 실패했다.
- 결과 해석: Transformer는 이번 설정에서 좋은 예측 구조라기보다 복잡한 shortcut을 더 쉽게 찾는 쪽으로 보인다. 데이터/정규화/손실 재설계 없이 바로 확장하면 위험하다.

#### 3.5.10 transformer_return_directional_hybrid
- 무엇을 시험했나: Transformer에 방향 벌점을 추가하면 Huber보다 shortcut이 줄어드는지 확인했다.
- 좋은 그림이면: 방향 벌점 때문에 sign agreement가 올라가고 persistence gap 악화가 없어야 한다.
- 이번 그림은 왜 그렇지 않은가: `persistence_gap = 1495041.468750`으로 더 나빠졌고, sign agreement도 `0.4156`으로 낮았다.
- 결과 해석: 현재 데이터 규모와 feature set에서는 Transformer + directional hybrid가 안정적이지 않다. 더 큰 구조를 쓰기 전에 stationarization과 regularization을 먼저 넣어야 한다.

#### 3.5.11 transformer_level_mse
- 무엇을 시험했나: 가장 강한 표현력의 구조로 가격 레벨을 직접 맞히면 복사형 함정을 이길 수 있는지 확인했다.
- 좋은 그림이면: level MSE라도 persistence gap이 빠르게 기준선 아래로 내려가고, gradient가 과도하게 커지지 않아야 한다.
- 이번 그림은 왜 그렇지 않은가: `persistence_gap = 105428593.343750`으로 단순 복사와 비교가 어려울 정도로 나빴고, 방향성도 약했다.
- 결과 해석: 가격 레벨 직접 회귀는 Transformer에서도 실패했다. 이 결과는 "복잡한 모델을 쓰면 해결된다"는 가정을 버리게 해준다.

#### 3.5.12 linear_level_mse
- 무엇을 시험했나: 가장 단순한 구조로 가격 레벨 직접 회귀가 얼마나 위험한지 확인하는 기준 케이스다.
- 좋은 그림이면: 단순 모델이라도 loss가 조금은 내려가고 persistence gap이 기준선 근처로 접근해야 한다.
- 이번 그림은 왜 그렇지 않은가: 손실이 거의 개선되지 않았고, `persistence_gap = 105427793.343750`으로 단순 복사보다 훨씬 못했다.
- 결과 해석: 가격 레벨을 직접 맞히는 설정 자체가 이번 데이터에서는 학습 신호를 제대로 만들지 못했다. 다음 단계에서는 level target을 기본 선택지에서 제외한다.

#### 3.5.13 lstm_level_mse
- 무엇을 시험했나: LSTM의 메모리 구조가 가격 레벨 직접 회귀의 한계를 보완하는지 확인했다.
- 좋은 그림이면: Linear보다 loss와 persistence gap이 뚜렷하게 좋아져야 한다.
- 이번 그림은 왜 그렇지 않은가: `persistence_gap = 105428601.343750`으로 Linear와 거의 같은 실패 패턴을 보였고, sign agreement도 낮았다.
- 결과 해석: 순환 메모리를 넣어도 target이 가격 레벨이면 문제가 그대로 남는다. 구조 변경보다 target 변경이 먼저다.

#### 3.5.14 gru_level_mse
- 무엇을 시험했나: 더 가벼운 순환 구조가 가격 레벨 직접 회귀를 조금이라도 안정화하는지 확인했다.
- 좋은 그림이면: LSTM보다 간단한 구조라도 gradient가 안정되고 persistence gap이 개선되어야 한다.
- 이번 그림은 왜 그렇지 않은가: `persistence_gap = 105428601.343750`으로 LSTM과 거의 같은 실패를 보였고, 방향성도 `0.4091`에 머물렀다.
- 결과 해석: GRU 역시 level MSE 문제를 해결하지 못했다. level target은 구조와 무관하게 가장 위험한 설정으로 분류한다.

#### 3.5.15 tcn_level_mse
- 무엇을 시험했나: TCN이 가격 레벨 직접 회귀에서도 다른 level MSE 구조보다 덜 무너지는지 확인했다.
- 좋은 그림이면: convolutional 구조가 국소 패턴을 잡아 persistence gap을 다른 level MSE보다 크게 줄여야 한다.
- 이번 그림은 왜 그렇지 않은가: `persistence_gap = 105428281.343750`으로 약간의 차이는 있어도 본질적으로 같은 실패다.
- 결과 해석: TCN의 장점은 return target과 directional hybrid에서만 일부 보였다. 가격 레벨 직접 회귀에서는 TCN도 shortcut 문제를 피하지 못했다.

## 4. 결론을 돕는 보조 그래프: collapse 진단

아래 그림은 학습 곡선 자체가 아니라, 모델이 쉬운 해로 붕괴하는지를 보조적으로 확인하기 위한 그림이다. 즉, loss가 줄어도 실전용 예측이 되는지 따로 확인하는 장치다.

노트북 출력 셀에 표시되는 `collapse_figure`를 보조적으로 확인한다.

### 4.1 보조 지표 설명

- `Collapse Score`: 여러 종류의 "쉬운 해 붕괴"를 한 번에 모아둔 경고 점수다. 예측이 너무 평평해지거나, 거의 0만 말하거나, 직전 값과 너무 비슷해질수록 올라간다.
- `Variance Ratio`: 실제 시장이 움직인 폭과, 모델이 예측한 움직임 폭을 비교한 값이다. 너무 작으면 모델이 지나치게 소심한 예측만 한다는 뜻이고, 너무 크면 실제보다 과장해서 흔들린다는 뜻이다.
- `Near-Zero Return Share`: 모델이 "다음 수익률은 거의 0일 것"이라고 답한 비율이다. 이 수치가 높으면 모델이 안전한 무변화 답으로 숨고 있을 가능성이 크다.
- `Persistence Gap`: 모델과 단순 복사를 직접 겨루게 한 점수다. 기준선 `0`보다 위면 단순 복사보다 못한 것이고, 아래로 내려가야 비로소 단순 복사를 이긴다.
- `Naive persistence`: "다음 값은 지금 값과 비슷할 것"이라고 답하는 아주 단순한 기준선이다. 이 기준선조차 못 넘으면 복잡한 딥러닝 모델을 쓸 이유가 약하다.

### 4.2 그래프가 왜 튀어 보이나

- 어떤 케이스는 손실만 보면 좋아 보이지만, 예측 폭을 너무 줄여 버리거나 단순 복사보다 여전히 못해서 실제론 쓸 수 없는 상태가 된다.
- 반대로 붕괴 점수는 낮아 보여도, 오를지 내릴지 맞히는 비율이 반반 수준이면 방향 신호로 쓰기 어렵다.
- 그래서 이 보조 그래프는 "학습이 되었나"보다 "그 학습이 제대로 된 예측으로 이어졌나"를 가려내기 위해 필요하다.

## 4.3 지표 요약

이 표는 마지막 epoch 기준으로 학습 곡선이 어디에 도달했는지 요약한다. 숫자 자체보다, 숫자가 '좋다/나쁘다'를 어떻게 말하는지 읽어야 한다.

| Case | Train loss index | Val loss index | Grad norm first->last | Collapse first->last | Persistence gap first->last |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_level_mse | 1.0043 | 1.0000 | 283430220.0000->86456683178.6667 | 0.0404->0.0513 | 116682439.6875->116681519.6875 |
| linear_return_huber | 0.0075 | 0.0575 | 0.3300->0.0313 | 0.1680->0.2043 | 1319811.6875->199394.6562 |
| linear_return_directional_hybrid | 0.4368 | 1.0925 | 0.6638->0.6866 | 0.1204->0.1839 | 2631759.9375->833003.8125 |
| lstm_level_mse | 1.0075 | 1.0000 | 267780649.3333->1288445322.6667 | 0.0285->0.0102 | 116682439.6875->116682423.6875 |
| lstm_return_huber | 0.0306 | 0.0311 | 0.4574->0.0417 | 0.1754->0.2244 | 1002588.4375->87366.3906 |
| lstm_return_directional_hybrid | 0.9399 | 0.9535 | 0.9162->0.0408 | 0.1689->0.2287 | 1750225.1875->109244.4688 |
| gru_level_mse | 1.0063 | 1.0000 | 305333498.6667->1312131253.3333 | 0.0496->0.0182 | 116682439.6875->116682423.6875 |
| gru_return_huber | 0.0067 | 0.0068 | 0.9874->0.0875 | 0.0415->0.2199 | 6271515.6875->159039.7969 |
| gru_return_directional_hybrid | 0.9034 | 0.9501 | 0.4792->0.0911 | 0.1382->0.2062 | 1739996.9375->204351.1250 |
| tcn_level_mse | 1.0077 | 1.0000 | 239370960.0000->48309649408.0000 | 0.0316->0.0211 | 116682439.6875->116682039.6875 |
| tcn_return_huber | 0.0021 | 0.0009 | 1.0469->0.0232 | 0.1977->0.5166 | 10714073.6875->21657.5547 |
| tcn_return_directional_hybrid | 0.7609 | 0.7747 | 1.0631->0.0554 | 0.1989->0.4354 | 6637008.1875->15259.9453 |
| transformer_level_mse | 1.0049 | 1.0000 | 1285879722.6667->2572240533.3333 | 0.0000->0.0000 | 116682439.6875->116682423.6875 |
| transformer_return_huber | 0.0536 | 0.0874 | 1.9405->3.6334 | 0.0249->0.1984 | 6446393.6875->1351746.4375 |
| transformer_return_directional_hybrid | 0.7118 | 0.7529 | 2.6845->2.9436 | 0.0500->0.1984 | 9080200.6875->1926584.5625 |

### 4.4 이 표를 어떻게 읽나

- `train loss index`와 `val loss index`가 둘 다 1 근처에 남아 있으면, 시작할 때와 비교해 거의 나아진 것이 없다고 보면 된다.
- `train loss index`는 많이 내려갔는데 `val loss index`는 덜 내려갔다면, 훈련 구간에서만 잘 맞추고 새 구간에서는 힘이 약하다는 뜻이다.
- `persistence gap`이 기준선 위에 있으면, 표가 아무리 복잡해 보여도 핵심 결론은 간단하다. 단순 복사보다 못한 것이다.
- `sign agreement`가 `0.5` 근처면 오를지 내릴지조차 뚜렷하게 못 맞힌다는 뜻이다.
- `collapse score`가 낮아도 위 조건들이 나쁘면 실전 후보로 올리기 어렵다.

## 4.5 현재 진단상 가장 덜 나쁜 후보

아래 후보는 `diagnostic_score` 기준의 상대 순위다. 이 값은 '좋다'는 뜻이 아니라 '덜 나쁘다'는 뜻이다. 이번 로컬 결과에서는 `tcn_return_directional_hybrid`만 test 기준 persistence gap을 소폭 음수로 만들었지만, 개선 폭이 매우 작기 때문에 여전히 확정 채택이 아니라 추가 검증 대상으로 본다.

- Case: `tcn_return_directional_hybrid`
- Algorithm: `tcn`
- Objective: `directional_hybrid`
- Diagnostic score: `0.8499`
- Collapse score: `0.5582`
- Persistence gap: `-10811.593750`
- Persistence ratio: `0.9338`
- Variance ratio: `0.3504`

중요: 이번 결과는 "완전 실패"와 "성공한 최종 모델 선정"의 중간쯤에 있다. 대부분 케이스는 여전히 `persistence_gap > 0`으로 baseline을 못 넘었고, 유일한 예외인 `tcn_return_directional_hybrid`도 개선 폭이 작아서 재현성 확인이 필요하다. 따라서 현재 해석은 `성공한 모델 확보`가 아니라 `어떤 구조와 목적함수가 shortcut 붕괴를 덜 허용하는지 확인한 중간 진단`이 맞다.

## 4.6 케이스별 상세 수치

### tcn_return_directional_hybrid
- 설명: Return regression plus direction penalty to test shortcut suppression quickly.
- 알고리즘 / 손실함수: `tcn` / `directional_hybrid`
- Diagnostic score: `0.8499`
- Collapse score: `0.5582`
- Persistence gap: `-10811.593750`
- Persistence ratio: `0.9338`
- Variance ratio: `0.3504`
- Near-zero return share: `0.1883`
- Sign agreement: `0.6234`

### lstm_return_directional_hybrid
- 설명: Return regression plus direction penalty to test shortcut suppression quickly.
- 알고리즘 / 손실함수: `lstm` / `directional_hybrid`
- Diagnostic score: `1.5454`
- Collapse score: `0.2181`
- Persistence gap: `85337.968750`
- Persistence ratio: `1.5224`
- Variance ratio: `1.1030`
- Near-zero return share: `0.0519`
- Sign agreement: `0.4740`

### lstm_return_huber
- 설명: Compact robust-return probe for a fast collapse check.
- 알고리즘 / 손실함수: `lstm` / `huber`
- Diagnostic score: `1.5765`
- Collapse score: `0.2204`
- Persistence gap: `95688.156250`
- Persistence ratio: `1.5858`
- Variance ratio: `1.2151`
- Near-zero return share: `0.0584`
- Sign agreement: `0.4610`

### gru_return_huber
- 설명: Compact robust-return probe for a fast collapse check.
- 알고리즘 / 손실함수: `gru` / `huber`
- Diagnostic score: `1.6639`
- Collapse score: `0.2134`
- Persistence gap: `154568.875000`
- Persistence ratio: `1.9462`
- Variance ratio: `1.8449`
- Near-zero return share: `0.0390`
- Sign agreement: `0.5455`

### gru_return_directional_hybrid
- 설명: Return regression plus direction penalty to test shortcut suppression quickly.
- 알고리즘 / 손실함수: `gru` / `directional_hybrid`
- Diagnostic score: `1.7356`
- Collapse score: `0.2043`
- Persistence gap: `183433.250000`
- Persistence ratio: `2.1229`
- Variance ratio: `2.1012`
- Near-zero return share: `0.0130`
- Sign agreement: `0.4351`

### linear_return_huber
- 설명: Compact robust-return probe for a fast collapse check.
- 알고리즘 / 손실함수: `linear` / `huber`
- Diagnostic score: `1.7972`
- Collapse score: `0.2133`
- Persistence gap: `208957.187500`
- Persistence ratio: `2.2792`
- Variance ratio: `2.1396`
- Near-zero return share: `0.0390`
- Sign agreement: `0.3961`

### tcn_return_huber
- 설명: Compact robust-return probe for a fast collapse check.
- 알고리즘 / 손실함수: `tcn` / `huber`
- Diagnostic score: `1.9511`
- Collapse score: `0.5649`
- Persistence gap: `23272.093750`
- Persistence ratio: `1.1425`
- Variance ratio: `0.3254`
- Near-zero return share: `0.1753`
- Sign agreement: `0.4091`

### linear_return_directional_hybrid
- 설명: Return regression plus direction penalty to test shortcut suppression quickly.
- 알고리즘 / 손실함수: `linear` / `directional_hybrid`
- Diagnostic score: `2.6418`
- Collapse score: `0.1993`
- Persistence gap: `857286.843750`
- Persistence ratio: `6.2481`
- Variance ratio: `6.8766`
- Near-zero return share: `0.0065`
- Sign agreement: `0.5519`

### transformer_return_huber
- 설명: Compact robust-return probe for a fast collapse check.
- 알고리즘 / 손실함수: `transformer` / `huber`
- Diagnostic score: `2.6996`
- Collapse score: `0.1998`
- Persistence gap: `997905.718750`
- Persistence ratio: `7.1090`
- Variance ratio: `2.1813`
- Near-zero return share: `0.0000`
- Sign agreement: `0.5844`

### transformer_return_directional_hybrid
- 설명: Return regression plus direction penalty to test shortcut suppression quickly.
- 알고리즘 / 손실함수: `transformer` / `directional_hybrid`
- Diagnostic score: `3.3407`
- Collapse score: `0.2020`
- Persistence gap: `1495041.468750`
- Persistence ratio: `10.1523`
- Variance ratio: `2.6231`
- Near-zero return share: `0.0065`
- Sign agreement: `0.4156`

### transformer_level_mse
- 설명: Direct next-close regression as the fastest copy-risk control case.
- 알고리즘 / 손실함수: `transformer` / `mse`
- Diagnostic score: `3.4366`
- Collapse score: `0.0248`
- Persistence gap: `105428593.343750`
- Persistence ratio: `646.4127`
- Variance ratio: `39.0142`
- Near-zero return share: `0.0000`
- Sign agreement: `0.4091`

### linear_level_mse
- 설명: Direct next-close regression as the fastest copy-risk control case.
- 알고리즘 / 손실함수: `linear` / `mse`
- Diagnostic score: `3.4536`
- Collapse score: `0.0418`
- Persistence gap: `105427793.343750`
- Persistence ratio: `646.4078`
- Variance ratio: `39.0137`
- Near-zero return share: `0.0000`
- Sign agreement: `0.4091`

### lstm_level_mse
- 설명: Direct next-close regression as the fastest copy-risk control case.
- 알고리즘 / 손실함수: `lstm` / `mse`
- Diagnostic score: `3.4938`
- Collapse score: `0.0820`
- Persistence gap: `105428601.343750`
- Persistence ratio: `646.4128`
- Variance ratio: `39.0142`
- Near-zero return share: `0.0000`
- Sign agreement: `0.4091`

### gru_level_mse
- 설명: Direct next-close regression as the fastest copy-risk control case.
- 알고리즘 / 손실함수: `gru` / `mse`
- Diagnostic score: `3.5049`
- Collapse score: `0.0931`
- Persistence gap: `105428601.343750`
- Persistence ratio: `646.4128`
- Variance ratio: `39.0142`
- Near-zero return share: `0.0000`
- Sign agreement: `0.4091`

### tcn_level_mse
- 설명: Direct next-close regression as the fastest copy-risk control case.
- 알고리즘 / 손실함수: `tcn` / `mse`
- Diagnostic score: `3.5206`
- Collapse score: `0.1087`
- Persistence gap: `105428281.343750`
- Persistence ratio: `646.4108`
- Variance ratio: `39.0138`
- Near-zero return share: `0.0000`
- Sign agreement: `0.4091`

## 5. 결론

이번 5번 실험 결론은 "loss가 줄었다"가 아니라 "무엇이 진짜 학습이고 무엇이 쉬운 해인지 구분할 기준이 잡혔다"에 가깝다. 이전 단계 그림에서 보였던 초반 급락 현상은 기울기 소실의 전형이라기보다, 비정상 가격 수준과 높은 자기상관이 결합된 상태에서 목적함수가 너무 쉬운 shortcut을 허용한 결과로 해석하는 편이 더 정확하다.

이번 로컬 결과를 5개 대표군 기준으로 정리하면 다음과 같다.

- `level_mse`: 5개 구조 전체에서 사실상 실패다. 가격 레벨 복사 위험을 제어하지 못했고, baseline 대비 우위도 전혀 보이지 않았다.
- `return_huber`: 학습은 가장 쉽게 진행됐지만, 그만큼 0 수익률 근처 수렴이나 축소된 분산 예측으로 흘러갈 위험도 분명했다.
- `directional_hybrid`: shortcut 억제 의도는 가장 잘 드러났고, 그중 `TCN`이 유일하게 test 기준 baseline을 소폭 넘었다.
- `Linear`, `LSTM`, `GRU`: 복잡도 차이는 있어도 현재 목적함수의 한계를 근본적으로 뒤집지는 못했다.
- `Transformer`: 표현력은 가장 강하지만, 이번 데이터 크기와 설계에서는 오히려 shortcut을 빠르게 찾는 쪽으로 작동한 흔적이 더 강했다.

따라서 지금 단계의 실무적 해석은 명확하다. 독립변수를 더 붙이기 전에 `return target 유지`, `directional penalty 재조정`, `variance collapse 방지 장치`, `baseline 대비 조기 종료/선택 기준`을 먼저 다듬어야 한다. 다시 말해 이번 보고서는 "최종 승자 발표"가 아니라, 이후 연구가 같은 함정으로 반복 붕괴하지 않도록 기준선을 세운 문서다.
