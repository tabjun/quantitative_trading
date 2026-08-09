# 8번 확장 학습 case 설계 근거

## 핵심 메시지

이번 문제는 “LSTM 대신 Transformer를 쓸 것인가” 같은 단순 알고리즘 선택 문제가 아니다. 같은 기본 모델 5개(`Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`)도 전처리, 정규화, 손실함수, optimizer, scheduler, gradient clipping, 앙상블 방식에 따라 loss landscape를 이동하는 방식이 달라진다. 따라서 8번은 모델 이름 확장이 아니라 **학습 경로 확장 실험**이어야 한다.

## 참고한 연구 흐름

### 1. 비정상 시계열 정규화

- Non-stationary Transformer는 실제 시계열의 분포가 시간에 따라 바뀌면 Transformer 성능이 크게 흔들릴 수 있고, 단순 stationarization이 오히려 중요한 비정상 정보를 지울 수 있다고 본다. 그래서 series stationarization과 de-stationary attention을 같이 쓴다.
- Dish-TS는 input window와 output horizon 사이에도 분포 shift가 생긴다고 보고, input-space와 output-space의 분포를 따로 추정하는 구조를 제안한다.
- FAN은 RevIN류가 평균/분산 같은 기본 trend는 다루지만 seasonal/frequency 성분에는 약하다고 보고, frequency component를 정규화 축으로 넣는다.
- NoRIN은 RevIN류 affine normalization이 heavy-tail과 skewness를 바꾸지 못한다고 지적하고, arcsinh 계열의 비선형 reversible normalization을 제안한다.

8번 반영:

- `window_standard`, `window_robust`, `window_minmax`
- `revin`, `robust_revin`, `asinh_revin`
- `asinh`, `ema_residual`, `frequency_highpass`

### 2. 단순 모델과 분해/패치 구조의 중요성

- DLinear/NLinear 계열은 복잡한 Transformer보다 단순 linear decomposition이 긴 시계열 예측에서 강할 수 있음을 보인다.
- PatchTST는 긴 sequence를 그대로 attention에 넣기보다 patch 단위 토큰화와 channel-independence가 계산량과 표현력을 동시에 개선할 수 있다고 본다.

8번 반영:

- `DLinearLike`, `NLinearLike`, `PatchTSTLike`
- 기본 5개 모델도 같은 preprocessing/normalization/loss/optimizer 조합으로 다시 확인

### 3. 손실함수와 평가 목표

- 금융/코인 시계열에서는 price-level MSE만 낮춰도 직전가 복사나 0 수익률 예측이 쉬운 해가 될 수 있다.
- quantile/pinball loss는 평균 하나가 아니라 조건부 분위수를 예측하는 관점에서 불확실성과 꼬리위험을 다룰 수 있다.
- 방향성 벌점은 단순 오차 최소화가 “크기는 맞지만 방향은 틀리는” 해를 만들 때 보조적으로 쓴다.

8번 반영:

- `return_mse`, `return_mae`
- `return_huber`
- `return_huber_directional`
- `vol_weighted_huber`
- `anti_collapse_huber`
- `quantile_pinball`

### 4. gradient와 optimizer 안정화

- Pascanu, Mikolov, Bengio는 RNN 학습에서 vanishing/exploding gradient를 분석하고, exploding gradient에 대한 실용적 해법으로 gradient norm clipping을 제안한다.
- Adam은 noisy/sparse gradient와 non-stationary objective에 적합한 adaptive optimizer로 널리 쓰인다.
- AdamW는 adaptive optimizer에서 weight decay와 L2 regularization이 같지 않다는 점을 분리해 일반화 성능을 개선하는 방향이다.
- SGDR/cosine restart 계열은 고정 learning rate보다 loss landscape 탐색을 더 다양하게 만들 수 있다.
- Lookahead는 inner optimizer의 빠른 weight와 slow weight를 섞어 학습 안정성과 variance를 낮추는 방향이다.

8번 반영:

- `adamw`, `adam`, `rmsprop`, `sgd_momentum`
- `cosine`, `onecycle`, `plateau`, `none`
- `clip0.5`, `clip1`, `clip5`, `adaptive`, `none`
- 학습 곡선에 `grad_norm_mean`, `grad_norm_max`, `lr` 저장

## 8번에서 봐야 할 질문

- 같은 `LSTM`이라도 `window_standard + return_huber`와 `asinh_revin + anti_collapse_huber`의 예측 분산이 다른가?
- 같은 `Transformer`라도 cosine scheduler와 plateau scheduler가 0 수익률 붕괴 정도를 다르게 만드는가?
- robust/asinh 계열 전처리가 extreme candle 이후 loss 폭주를 줄이는가?
- frequency/highpass 전처리가 직전가 복사형 baseline과의 중첩을 줄이는가?
- gradient norm이 초반부터 거의 0으로 죽거나 반대로 튀는 case가 특정 모델/손실/정규화에 몰리는가?
- 앙상블은 단순히 MAE만 낮추는가, 아니면 collapse 지표도 같이 낮추는가?

## 주요 참고 링크

- Non-stationary Transformers: https://arxiv.org/abs/2205.14415
- Dish-TS: https://arxiv.org/abs/2302.14829
- FAN: https://arxiv.org/abs/2409.20371
- NoRIN: https://arxiv.org/abs/2605.10823
- DLinear/NLinear: https://arxiv.org/abs/2205.13504
- PatchTST: https://arxiv.org/abs/2211.14730
- Gradient clipping for RNNs: https://arxiv.org/abs/1211.5063
- Adam: https://arxiv.org/abs/1412.6980
- AdamW: https://arxiv.org/abs/1711.05101
- SGDR: https://arxiv.org/abs/1608.03983
- Lookahead: https://arxiv.org/abs/1907.08610

