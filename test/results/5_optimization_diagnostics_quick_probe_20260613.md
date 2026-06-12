# 5. 최적화 학습 과정 진단 보고서

## 1. 한눈에 보는 결론

- 이번 quick probe에서 현재 가장 덜 나쁜 후보는 `lstm_return_huber` 이지만, 이것은 최종 채택이 아니라 상대적 우위일 뿐이다.
- 핵심 문제는 `persistence_gap`이 아직 양수라는 점이다. 즉, 현재 실험 세팅에서는 단순 직전가 복사 baseline을 안정적으로 이기지 못했다.
- loss curve는 일부 개선됐지만, validation이 충분히 같이 내려가지 않거나 그래프별 보조 신호가 서로 엇갈려서, 아직 목적함수와 target 정의가 완전히 맞물린 상태는 아니다.
- 방향성(`sign agreement`)도 대체로 동전 던지기 근처라, 현재 결과를 바로 트레이딩 신호로 해석하면 안 된다.

### 1.1 지금 이 결과를 어떻게 읽어야 하나

- train loss가 내려가는 것은 '최소한 배운다'는 뜻이지 '좋은 예측을 한다'는 뜻은 아니다.
- validation이 같이 내려가야 다음 구간 일반화 신호로 볼 수 있다.
- `persistence_gap > 0`이면 복잡한 모델이 직전가 복사보다 못한 것이다.
- `collapse_score`가 낮아도 baseline을 못 이기면 아직 연구용 probe 단계다.
- `near-zero return share`가 낮아도 방향성이 약하면 경제적 가치가 낮다.

### 1.2 케이스별 짧은 해석

- `[lstm_level_mse]` 최종 판단: loss 곡선이 평평해서 현재 objective가 유효한 학습 신호를 충분히 주지 못한다. / gradient가 아직 큰 편이라 objective scale 또는 target scale이 다소 거칠 수 있다. / 직전가 복사 baseline보다 아직 못하다.
- `[lstm_level_mse]` 최종 수치: train loss index `0.9921`, val loss index `1.0000`, gap `0.0079`
- `[lstm_level_mse]` 보조 신호: grad `1288588810.6667`, collapse `0.0333`, zero share `0.0000`, persistence gap `116682423.6875`, sign `0.4935`

- `[lstm_return_huber]` 최종 판단: train과 validation이 같이 내려가서 최적화 자체는 진행되고 있다. / gradient가 너무 작아져 학습 신호가 약한 상태일 수 있다. / 직전가 복사 baseline보다 아직 못하다.
- `[lstm_return_huber]` 최종 수치: train loss index `0.0082`, val loss index `0.0063`, gap `-0.0019`
- `[lstm_return_huber]` 보조 신호: grad `0.0333`, collapse `0.2286`, zero share `0.0844`, persistence gap `94434.2812`, sign `0.4740`

- `[lstm_return_directional_hybrid]` 최종 판단: train과 validation이 같이 내려가서 최적화 자체는 진행되고 있다. / gradient가 너무 작아져 학습 신호가 약한 상태일 수 있다. / 직전가 복사 baseline보다 아직 못하다.
- `[lstm_return_directional_hybrid]` 최종 수치: train loss index `0.8114`, val loss index `0.8766`, gap `0.0652`
- `[lstm_return_directional_hybrid]` 보조 신호: grad `0.0626`, collapse `0.2125`, zero share `0.0390`, persistence gap `127384.7812`, sign `0.5260`

### 1.3 교수님께 한 문장으로 말하면

- 이번 결과는 일부 objective가 학습 신호를 받긴 했지만, validation 일반화와 persistence baseline 돌파가 아직 충분하지 않아 최종 모델 선정보다는 objective/target 재조정이 먼저라는 뜻이다.

## 2. 목적

이 실험은 모델 성능 순위를 뽑는 리더보드가 아니라, 손실함수와 모델 구조가 실제로 학습되고 있는지 확인하기 위한 최적화 진단 실험이다.
특히 비정상 금융 시계열에서 모델이 가장 쉬운 해인 `0 수익률 예측`, `직전 가격 복사`, `validation 정체`로 붕괴하는지 확인하는 것이 목적이다.
따라서 이 보고서는 RMSE 하나로 모델을 고르는 문서가 아니라, 학습 곡선과 최적화 상태를 먼저 읽고 이후 collapse 지표를 보조적으로 해석하는 문서이다.

## 3. 실행 설정

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

## 4. 방법론 및 진단 지표 정의

이 절은 결과를 보기 전에 각 지표가 무엇을 의미하는지 정의한다. 본 실험의 목적은 최종 수익률을 주장하는 것이 아니라, 손실함수와 모델 구조가 비정상 금융 시계열에서 어떤 방식으로 최적화되는지 진단하는 것이다.

### 4.1 Train/Validation Loss Curve

- 개념: epoch가 진행될 때 학습 데이터 손실(train loss)과 검증 데이터 손실(validation loss)이 어떻게 움직이는지 보는 기본 최적화 곡선이다.
- 사용 이유: 모델이 단순히 train 데이터만 외우는지, 아니면 보지 않은 validation 구간에서도 손실이 줄어드는지 확인하기 위해 사용한다.
- 정의: `train_loss = L(y_train, yhat_train)`, `validation_loss = L(y_val, yhat_val)`이며, 본 실험에서는 서로 다른 손실함수 스케일을 비교하기 위해 첫 epoch 값을 1.0으로 둔 loss index도 함께 본다.
- 해석 예시: train과 validation이 함께 하락하면 학습이 진행되는 것이다. train만 하락하고 validation이 정체되면 과적합 또는 쉬운 해로의 붕괴 가능성이 있다.
- 장점: 가장 직관적으로 최적화 과정의 수렴, 정체, 발산을 확인할 수 있다.
- 한계: 금융 시계열에서는 validation loss가 낮아도 방향성 또는 매매 가능성이 보장되지 않는다. 따라서 persistence gap, sign agreement 같은 보조 지표가 필요하다.

### 4.2 Generalization Gap

- 개념: validation loss와 train loss 사이의 차이다.
- 사용 이유: 학습 데이터에서는 좋아지지만 미래 구간에서는 좋아지지 않는 일반화 실패를 확인하기 위해 사용한다.
- 정의: `gap = validation_loss_index - train_loss_index`.
- 해석 예시: gap이 양수로 커지면 train 데이터에는 적응하지만 validation 구간에는 통하지 않는다는 뜻이다.
- 장점: 과적합 여부를 빠르게 볼 수 있다.
- 한계: train/validation 구간 자체가 regime shift를 포함하면 gap은 과적합뿐 아니라 시장 국면 변화의 신호일 수도 있다.

### 4.3 Gradient Norm Before Clipping

- 개념: 역전파 후 gradient clipping을 적용하기 직전의 gradient 크기다.
- 사용 이유: 손실함수 스케일이 너무 크거나 모델이 불안정해서 학습이 튀는지 확인하기 위해 사용한다.
- 정의: `||grad|| = sqrt(sum_i grad_i^2)`.
- 해석 예시: gradient norm이 계속 clipping 기준에 붙으면 손실 스케일 또는 target 스케일이 너무 공격적일 수 있다. 반대로 너무 작으면 학습 신호가 약하거나 평평한 해에 갇혔을 수 있다.
- 장점: loss curve만으로 보이지 않는 최적화 안정성을 확인할 수 있다.
- 한계: gradient norm이 안정적이라고 해서 예측력이 좋다는 뜻은 아니다. 안정적으로 잘못된 해에 수렴할 수도 있다.

### 4.4 Persistence Gap

- 개념: 모델의 절대오차가 단순 직전가 복사 기준보다 얼마나 나은지 보는 baseline 비교 지표다.
- 사용 이유: 금융 가격 레벨은 자기상관이 매우 강해서, 복잡한 모델이 실제로는 직전 가격을 베끼는 것보다 못한 경우가 많다.
- 정의: `persistence_gap = MAE(model_prediction, y_next) - MAE(current_price, y_next)`.
- 해석 예시: 값이 0보다 작으면 모델이 naive persistence보다 낫다. 0보다 크면 복잡한 모델을 썼지만 직전가 복사보다 못하다는 뜻이다.
- 장점: 가격 레벨 예측에서 발생하는 기만적 RMSE 착시를 방지한다.
- 한계: persistence baseline은 가격 레벨 기준이다. 방향성 전략이나 변동성 예측에서는 별도의 경제적 baseline도 함께 봐야 한다.

### 4.5 Collapse Score

- 개념: 모델이 예측 변동성을 잃거나, 0 수익률만 내거나, 직전 가격과 과도하게 붙는 쉬운 해로 도망가는 위험을 합친 보조 점수다.
- 사용 이유: 손실함수 최적화가 허용하는 가장 쉬운 해가 실제 예측이 아니라 평균/무변화/복사일 수 있기 때문이다.
- 정의: 본 코드에서는 낮은 prediction variance, near-zero return share, copy alignment를 가중 결합한다.
- 해석 예시: 낮은 collapse score는 붕괴 위험이 작다는 뜻일 수 있지만, persistence gap이 양수라면 여전히 예측 모델로 채택하면 안 된다.
- 장점: 모델이 명백히 평평한 예측 또는 복사 예측으로 무너지는지 빠르게 확인할 수 있다.
- 한계: 단일 성능 지표가 아니다. collapse score만 낮다고 좋은 모델이 아니다.

### 4.6 Near-Zero Return Share

- 개념: 모델이 예측한 다음 수익률이 거의 0에 가까운 비율이다.
- 사용 이유: 수익률 예측 모델이 '다음에도 거의 안 움직인다'는 평균 해로 붕괴하는지 확인하기 위해 사용한다.
- 정의: `mean(abs(predicted_return) < 1e-4)`.
- 해석 예시: 이 값이 높으면 모델이 가격 변동을 예측하기보다 무변화 예측으로 손실을 낮추고 있을 가능성이 있다.
- 장점: 수익률 target에서 평균 수렴/무변화 예측을 탐지하기 쉽다.
- 한계: 실제 시장이 저변동 구간이면 near-zero 비율이 높게 나올 수 있으므로, realized volatility와 함께 해석해야 한다.

### 4.7 Sign Agreement

- 개념: 예측한 가격 변화 방향과 실제 가격 변화 방향의 부호가 일치한 비율이다.
- 사용 이유: 가격 오차가 작아도 상승/하락 방향을 못 맞히면 트레이딩 관점에서 유효성이 낮기 때문이다.
- 정의: `mean(sign(predicted_delta) == sign(actual_delta))`.
- 해석 예시: 0.5 근처면 동전 던지기 수준이다. 유의미한 매매 신호로 보려면 수수료와 슬리피지를 넘는 방향성 우위가 필요하다.
- 장점: 가격 레벨 RMSE의 착시를 보완한다.
- 한계: 방향만 맞고 크기를 틀리면 실제 손익은 나쁠 수 있다.

### 4.8 알고리즘 가족 설명

- `Linear`: 입력 윈도우를 한 번에 펴서 가장 단순한 선형 조합으로 예측한다. 장점은 빠르고 해석이 쉬우며, 단점은 시계열 순서 정보를 거의 직접 쓰지 못한다는 점이다.
- `LSTM`: 과거 상태를 게이트로 누적하는 순환 구조다. 장점은 시계열의 순서를 기억할 수 있다는 점이고, 단점은 데이터가 적거나 objective가 거칠면 쉽게 평평한 해로 수렴할 수 있다는 점이다.
- `GRU`: LSTM보다 게이트가 단순한 순환 구조다. 장점은 더 가볍고 빠르게 학습된다는 점이며, 단점은 경우에 따라 복잡한 장기 의존성을 덜 잡을 수 있다는 점이다.
- 이번 실험에서 `Linear`/`LSTM`/`GRU`를 함께 두는 이유는, 복잡도가 높을수록 항상 좋은 것이 아니라는 점을 확인하기 위해서다. 같은 objective에서도 아키텍처가 다르면 붕괴 방식이 달라질 수 있다.

## 5. 먼저 볼 그래프: 최적화 학습 곡선

아래 그림은 전체 케이스를 한 번에 비교하기 위한 요약 그림이다. 개별 모델별 상세 학습 곡선은 다음 절에 따로 제공한다.

노트북 출력 셀에 표시되는 `training_figure`를 먼저 확인한다.

![전체 최적화 학습 곡선](../images/5_optimization_diagnostics_test_cell003_01.png)

이 그림은 전체 케이스의 학습 안정성, 일반화 갭, gradient 안정성, persistence baseline 대비 위치를 한 번에 본다.

### 5.1 그래프별 축과 해석 기준

- `Train/Validation Objective Loss Index`: x축은 epoch, y축은 1번째 epoch의 손실을 1.0으로 둔 상대 손실이다. 왼쪽 위 패널에서 본다. 이 값이 함께 내려가야 학습이 실제로 진행된다고 해석한다.
- `Validation - Train Gap`: x축은 epoch, y축은 validation 손실 지수에서 train 손실 지수를 뺀 값이다. 오른쪽 위 패널에서 본다. 양수로 벌어지면 train만 외우고 validation은 개선하지 못하는 과적합 또는 일반화 실패 신호다.
- `Gradient Norm Before Clipping`: x축은 epoch, y축은 gradient clipping 직전의 평균 gradient norm이다. 왼쪽 아래 패널에서 본다. 계속 clipping 기준선에 붙거나 크게 튀면 손실 스케일 또는 objective가 불안정하다는 의미다.
- `Validation Persistence Gap`: x축은 epoch, y축은 `모델 MAE - 직전가 복사 naive MAE`이다. 오른쪽 아래 패널에서 본다. 0 아래로 내려가야 naive copy보다 낫다는 뜻이다.
- 왼쪽 위는 학습이 되고 있는가를, 오른쪽 위는 generalization이 되는가를, 왼쪽 아래는 최적화가 안정적인가를, 오른쪽 아래는 baseline을 이기는가를 본다고 기억하면 된다.

### 5.2 현재 결과를 읽는 핵심 기준

- train loss만 내려가고 validation loss가 평평하면, 모델은 학습 데이터에는 반응하지만 미래 구간 일반화에는 실패한 것이다.
- validation loss가 거의 움직이지 않으면, 현재 손실함수/입력 구성이 가격 변동의 유효한 곡률을 제공하지 못한다는 뜻이다.
- persistence gap이 계속 양수이면, 모델이 복잡한 신경망임에도 단순 직전가 복사보다 예측력이 낮다.
- 따라서 이 실험에서는 `loss가 낮다`보다 `validation이 내려가는가`, `gap이 줄어드는가`, `persistence gap이 0 근처로 가는가`를 우선한다.

### 5.3 현재 그래프가 애매하게 보일 때의 해석

- loss는 내려가는데 다른 그래프가 튀면, 모델이 최적화는 하고 있지만 그 신호가 안정적으로 예측력으로 이어지지 않는 상태다.
- collapse score나 zero share가 좋아 보여도 persistence gap이 양수면, 여전히 baseline보다 못한 학습이다.
- sign agreement가 0.5 부근이면 방향성은 아직 약해서, 값의 크기보다 방향만 보는 전략도 조심해야 한다.
- gradient norm이 들쑥날쑥하면 objective scale이 거칠거나 입력 스케일이 아직 충분히 정돈되지 않았을 수 있다.

## 6. 모델별 학습 곡선

각 모델/손실함수 조합을 분리해서 본 그림이다. 이전 2번 실험의 모델별 loss curve처럼, 여기서는 최적화 과정만 빠르게 확인하기 위해 케이스별로 분리했다.
아래 각 패널은 모두 같은 읽는 법을 따른다. 왼쪽 위는 loss가 내려가는지, 오른쪽 위는 train과 validation의 차이가 커지는지, 왼쪽 아래는 gradient가 과하게 튀는지, 오른쪽 아래는 직전가 복사 baseline을 이기는지 본다.

### 6.1 lstm_return_huber

![lstm_return_huber 학습 곡선](../images/5_optimization_diagnostics_test_cell003_03.png)

이 케이스는 return target에 Huber loss를 적용했을 때 학습 곡선이 얼마나 안정적으로 내려가는지 보여준다.
- 알고리즘 해설: LSTM은 과거 상태를 기억하는 능력이 있어 시계열 정보를 더 잘 쓸 수 있지만, 데이터가 적거나 loss가 거칠면 쉽게 평평한 해로 수렴할 수 있다.
- 손실함수 해설: Huber는 큰 outlier에 덜 민감해서, 비정상 구간에서 학습을 조금 더 부드럽게 만들 수 있다.
이 케이스의 그림은 노트북 출력 셀의 모델별 figure로 표시된다.

- 왼쪽 위: train/validation loss가 함께 하락하는지 확인한다.
- 오른쪽 위: validation과 train의 차이가 커지는지 확인한다.
- 왼쪽 아래: gradient가 안정적으로 흐르는지 확인한다.
- 오른쪽 아래: 모델이 naive persistence를 이기는 방향으로 가는지 확인한다.
- 현재 test 기준 persistence gap: `91703.421875`
- 현재 sign agreement: `0.5130`

### 6.2 lstm_return_directional_hybrid

![lstm_return_directional_hybrid 학습 곡선](../images/5_optimization_diagnostics_test_cell003_04.png)

이 케이스는 값 예측과 방향성 penalty를 함께 둘 때 최적화가 어떻게 달라지는지 보여준다.
- 알고리즘 해설: LSTM은 과거 상태를 기억하는 능력이 있어 시계열 정보를 더 잘 쓸 수 있지만, 데이터가 적거나 loss가 거칠면 쉽게 평평한 해로 수렴할 수 있다.
- 손실함수 해설: directional hybrid는 값 예측에 방향성 penalty를 얹어, 0 수익률로 도망가는 쉬운 해를 줄이려는 설계다.
이 케이스의 그림은 노트북 출력 셀의 모델별 figure로 표시된다.

- 왼쪽 위: train/validation loss가 함께 하락하는지 확인한다.
- 오른쪽 위: validation과 train의 차이가 커지는지 확인한다.
- 왼쪽 아래: gradient가 안정적으로 흐르는지 확인한다.
- 오른쪽 아래: 모델이 naive persistence를 이기는 방향으로 가는지 확인한다.
- 현재 test 기준 persistence gap: `111651.218750`
- 현재 sign agreement: `0.5000`

### 6.3 lstm_level_mse

![lstm_level_mse 학습 곡선](../images/5_optimization_diagnostics_test_cell003_02.png)

이 케이스는 가격 레벨을 직접 맞추는 통제군으로서, 복사형 shortcut 위험을 가장 강하게 드러낸다.
- 알고리즘 해설: LSTM은 과거 상태를 기억하는 능력이 있어 시계열 정보를 더 잘 쓸 수 있지만, 데이터가 적거나 loss가 거칠면 쉽게 평평한 해로 수렴할 수 있다.
- 손실함수 해설: MSE는 큰 오차를 강하게 벌주므로, 가격 레벨 회귀에서는 복사형 해를 밀어낼 수도 있지만 scale에 민감하다.
이 케이스의 그림은 노트북 출력 셀의 모델별 figure로 표시된다.

- 왼쪽 위: train/validation loss가 함께 하락하는지 확인한다.
- 오른쪽 위: validation과 train의 차이가 커지는지 확인한다.
- 왼쪽 아래: gradient가 안정적으로 흐르는지 확인한다.
- 오른쪽 아래: 모델이 naive persistence를 이기는 방향으로 가는지 확인한다.
- 현재 test 기준 persistence gap: `105428601.343750`
- 현재 sign agreement: `0.4091`

## 7. 보조 그래프: collapse 진단

아래 그림은 학습 곡선 자체가 아니라, 모델이 쉬운 해로 붕괴하는지를 보조적으로 확인하기 위한 그림이다.

노트북 출력 셀에 표시되는 `collapse_figure`를 보조적으로 확인한다.

![붕괴 진단 요약](../images/5_optimization_diagnostics_test_cell003_05.png)

이 그림은 loss만 줄어드는 것처럼 보여도 실제로는 prediction variance가 사라지거나 near-zero return으로 도망치는지 함께 본다.

### 7.1 보조 지표 설명

- `Collapse Score`: 예측 변동성이 사라지거나, 0 수익률만 내거나, 직전 가격과 지나치게 붙는 현상을 합친 붕괴 위험 점수다. 낮을수록 좋지만 단독으로 모델을 고르면 안 된다.
- `Variance Ratio`: 실제 가격 변화 표준편차 대비 예측 가격 변화 표준편차다. 0에 가까우면 평평한 예측으로 붕괴한 것이고, 너무 크면 과도하게 흔들리는 예측이다.
- `Near-Zero Return Share`: 예측한 다음 수익률이 거의 0에 가까운 비율이다. 높으면 모델이 '다음에도 거의 안 움직인다'는 쉬운 답으로 도망간 것이다.
- `Persistence Gap`: 이 값이 양수이면 모델이 단순 직전가 복사보다 못하다는 뜻이다. 이 실험에서는 가장 중요한 방어선이다.

### 7.2 그래프가 왜 튀어 보이나

- 어떤 케이스는 loss가 줄어들어도 variance_ratio가 높거나 zero_share가 낮아서 '겉보기엔 괜찮지만 실제론 baseline을 못 넘는' 상태가 된다.
- 반대로 collapse score는 낮지만 sign agreement가 0.5 근처면, 무너진 예측은 아니어도 방향 신호가 약하다.
- 따라서 세 그래프를 같이 봐야 한다: loss는 학습 여부, collapse는 쉬운 해 여부, sign agreement는 방향성 여부를 말한다.

## 8. 지표 요약

이 표는 마지막 epoch 기준으로 학습 곡선이 어디에 도달했는지 요약한다.

| Case | Train loss index | Val loss index | Grad norm first->last | Collapse first->last | Persistence gap first->last |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstm_level_mse | 0.9921 | 1.0000 | 272970144.0000->1288588810.6667 | 0.0248->0.0333 | 116682439.6875->116682423.6875 |
| lstm_return_huber | 0.0082 | 0.0063 | 0.6726->0.0333 | 0.1365->0.2286 | 3674269.1875->94434.2812 |
| lstm_return_directional_hybrid | 0.8114 | 0.8766 | 1.1334->0.0626 | 0.1301->0.2125 | 3797656.1875->127384.7812 |

## 9. 현재 진단상 가장 덜 나쁜 후보

아래 후보는 `diagnostic_score` 기준의 상대 순위다. 단, 모든 케이스의 persistence gap이 양수이면 최종 채택이 아니라 추가 수정 대상으로만 본다.

- Case: `lstm_return_huber`
- Algorithm: `lstm`
- Objective: `huber`
- Diagnostic score: `1.5574`
- Collapse score: `0.2090`
- Persistence gap: `91703.421875`
- Persistence ratio: `1.5614`
- Variance ratio: `1.4360`

중요: 모든 케이스에서 `persistence_gap > 0`이면, 이 실험은 '성공한 모델 선정'이 아니라 '현재 목적함수/입력 구성으로는 아직 최적화가 부족하다'는 진단으로 해석한다.

## 10. 케이스별 상세 수치

### lstm_return_huber
- 설명: Compact robust-return probe for a fast collapse check.
- 알고리즘 / 손실함수: `lstm` / `huber`
- Diagnostic score: `1.5574`
- Collapse score: `0.2090`
- Persistence gap: `91703.421875`
- Persistence ratio: `1.5614`
- Variance ratio: `1.4360`
- Near-zero return share: `0.0260`
- Sign agreement: `0.5130`

### lstm_return_directional_hybrid
- 설명: Return regression plus direction penalty to test shortcut suppression quickly.
- 알고리즘 / 손실함수: `lstm` / `directional_hybrid`
- Diagnostic score: `1.5901`
- Collapse score: `0.2067`
- Persistence gap: `111651.218750`
- Persistence ratio: `1.6835`
- Variance ratio: `1.5957`
- Near-zero return share: `0.0195`
- Sign agreement: `0.5000`

### lstm_level_mse
- 설명: Direct next-close regression as the fastest copy-risk control case.
- 알고리즘 / 손실함수: `lstm` / `mse`
- Diagnostic score: `3.5048`
- Collapse score: `0.0930`
- Persistence gap: `105428601.343750`
- Persistence ratio: `646.4128`
- Variance ratio: `39.0142`
- Near-zero return share: `0.0000`
- Sign agreement: `0.4091`

## 11. 결론 및 다음 실행 기준

이번 5번 실험은 모델의 최종 성능보다 최적화 과정이 정상적으로 움직이는지 확인하는 단계다.
따라서 다음 실행에서는 모델별 학습 곡선에서 validation loss가 실제로 하락하는지, gradient가 안정적인지, persistence gap이 0으로 접근하는지를 먼저 확인해야 한다.
만약 1년 데이터와 30 epoch에서도 validation이 평평하고 persistence gap이 양수라면, 단순히 독립변수를 늘리기보다 target 정의, loss scaling, baseline 대비 objective를 먼저 수정해야 한다.
