# 8번 최적화 breadth training 결과 보고서

## 실험 설계

> 실행 전에 쓴 계획서·설계 문서 원문이다(2026-10-14 `test/experiment_specs/` 폐지 때 이 보고서로 합쳤다. 원 파일은 git 이력에 남아 있다). 제목 수준을 두 단계 내리고 옮겨진 파일의 경로만 새 위치로 고쳤으며, 그 밖의 내용은 고치지 않았다.

> 원문: 구 `test/experiment_specs/8_optimization_breadth_training_plan_20260616.md`

### 8번 최적화 breadth training 설계

#### 역할

8번은 7번의 stage plan을 실제 학습 실험으로 바꾸는 후속 번호다. 5번은 shortcut collapse 진단 기준선, 6번은 안정화 기준 정리, 7번은 자원 인식형 확장 계획이었다. 8번은 이 계보를 보존하면서 실제 GPU 학습, 시각화, 성능표, collapse 진단을 생성한다.

중요한 점은 8번이 단순히 알고리즘 이름만 늘리는 실험이 아니라는 것이다. 같은 5개 기본 모델(`Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`)도 전처리, 정규화, 손실함수, optimizer, scheduler, gradient clipping, 앙상블 방식에 따라 학습 경로가 달라질 수 있다. 따라서 8번은 **알고리즘 breadth**와 **학습 방식 breadth**를 함께 본다.

#### 왜 8번이 필요한가

7번 결과는 모델 성능 결과가 아니라 계획 출력이었다. 따라서 “Autoformer-like가 실제로 개선되는가”, “PatchTST-like가 0 수익률 붕괴를 줄이는가”, “DLinear/NLinear-like 같은 단순 구조가 복잡한 Transformer보다 안정적인가”, “앙상블이 직전가 복사 문제를 줄이는가”를 판단할 수 없었다.

또한 7번은 같은 모델의 학습 방식을 바꾸는 실험을 충분히 포함하지 않았다. 하지만 비정상 업비트 시계열에서는 모델 구조보다 preprocessing, normalization, loss, gradient flow가 더 큰 영향을 줄 수 있다. 그러므로 8번은 모델 family 확장과 함께 학습 조건 확장을 동시에 수행한다.

#### 고정할 안정화 기준

- Target: 다음 가격 레벨이 아니라 다음 로그수익률
- Evaluation: 예측 로그수익률을 KRW 가격으로 복원한 MAE/RMSE
- Preprocessing: `none`, `winsorize`, `asinh`, `diff`, `ema_residual`, `frequency_highpass`
- Normalization: `window_standard`, `window_robust`, `window_minmax`, `revin`, `robust_revin`, `asinh_revin`, `global_standard`
- Loss: `return_mse`, `return_mae`, `return_huber`, `return_huber_directional`, `vol_weighted_huber`, `anti_collapse_huber`, `quantile_pinball`
- Optimizer/Scheduler: `adamw`, `adam`, `rmsprop`, `sgd_momentum`와 `cosine`, `onecycle`, `plateau`, `none`
- Gradient policy: `clip0.5`, `clip1`, `clip5`, `adaptive`, `none`
- Model selection: validation loss만 보지 않고 persistence gap, variance ratio, near-zero return share, copy-risk ratio를 함께 확인

#### 포함 모델군

기본 대표군:

- Linear
- LSTM
- GRU
- TCN
- Transformer

확장 단일 모델군:

- AutoformerLike
- PatchTSTLike
- DLinearLike
- NLinearLike
- TimesNetLike
- TimeXerLike
- ITransformerLike
- ModernTCNLike
- MambaLike

앙상블군:

- LSTM + Autoformer mean
- TCN + Transformer mean
- Linear + sequence residual-style mean
- validation weighted top-k ensemble

#### 추가 suite

- `breadth_probe`: 기본 안정화 조건에서 모델 family 자체를 넓게 비교한다.
- `core_tuning_probe`: 6번의 5개 기본 모델을 유지하되 전처리, 정규화, 손실함수, optimizer, scheduler, gradient clipping을 함께 흔들어 본다.
- `preprocessing_cross_check`: 모델을 고정하고 `winsorize`, `asinh`, `diff`, `ema_residual`, `frequency_highpass`가 collapse 지표에 미치는 영향을 본다.
- `normalization_cross_check`: `window_standard`, `window_robust`, `window_minmax`, `revin`, `robust_revin`, `asinh_revin`을 비교한다.
- `loss_cross_check`: Huber, 방향성 벌점, 변동성 가중, anti-collapse, pinball loss를 비교한다.
- `optimization_cross_check`: optimizer와 scheduler 조합이 같은 모델의 학습 경로를 어떻게 바꾸는지 본다.
- `gradient_cross_check`: gradient clipping 강도와 adaptive clipping이 loss/gradient norm/붕괴 지표를 어떻게 바꾸는지 본다.
- `ensemble_probe`: 단일 모델 결과를 묶어 앙상블이 collapse와 KRW MAE를 동시에 줄이는지 본다.
- `full`: 넓은 조합을 순차적으로 확인하되, 서버 시간이 길어질 수 있으므로 중간 저장 대신 notebook inline 출력과 단계별 재실행을 전제로 한다.

#### 서버 자원 기준

기본 profile은 `school_4090_15gb`다.

- device: cuda 우선
- n_jobs: 16 이하
- max_workers: 16 이하
- DataLoader num_workers: 4
- torch_num_threads: 16 이하
- torch_interop_threads: 4
- optuna_n_jobs: GPU 학습에서는 1
- batch_size: 기본 48, OOM 발생 시 자동 절반 축소
- max_cases: 기본 120개로 제한, 전체 실행은 `--max-cases 0`
- 중간 결과는 case마다 notebook 출력으로 확인하고, 파일 저장은 기본적으로 하지 않는다.

#### 실행 예시

작게 확인:

```bash
python test/models/8_optimization_breadth_training_test.py \
  --suite breadth_probe \
  --models Linear,LSTM,TCN,Transformer,AutoformerLike,PatchTSTLike \
  --epochs 3 \
  --max-windows 1024 \
  --batch-size 48 \
  --continue-on-failure
```

기본 5개 모델의 학습 방식 튜닝 비교:

```bash
python test/models/8_optimization_breadth_training_test.py \
  --suite core_tuning_probe \
  --epochs 6 \
  --max-windows 2048 \
  --max-cases 120 \
  --continue-on-failure
```

전처리 비교:

```bash
python test/models/8_optimization_breadth_training_test.py \
  --suite preprocessing_cross_check \
  --epochs 8 \
  --max-windows 2048 \
  --continue-on-failure
```

loss 비교:

```bash
python test/models/8_optimization_breadth_training_test.py \
  --suite loss_cross_check \
  --epochs 8 \
  --max-windows 2048 \
  --continue-on-failure
```

gradient clipping 비교:

```bash
python test/models/8_optimization_breadth_training_test.py \
  --suite gradient_cross_check \
  --epochs 8 \
  --max-windows 2048 \
  --continue-on-failure
```

앙상블 확인:

```bash
python test/models/8_optimization_breadth_training_test.py \
  --suite ensemble_probe \
  --epochs 8 \
  --max-windows 2048 \
  --continue-on-failure
```

#### 산출물

- `test/results/8_optimization_breadth_training_<timestamp>_summary.csv`
- `test/results/8_optimization_breadth_training_<timestamp>_curves.csv`
- `test/results/8_optimization_breadth_training_<timestamp>_report.md`
- `test/images/8_optimization_breadth_training_<timestamp>_*_learning_curve.png`
- `test/images/8_optimization_breadth_training_<timestamp>_*_prediction.png`
- `test/images/8_optimization_breadth_training_<timestamp>_collapse_diagnostics.png`

`curves.csv`에는 `train_loss`, `val_loss`뿐 아니라 `grad_norm_mean`, `grad_norm_max`, `lr`도 저장한다. 따라서 loss만 줄어드는지, gradient가 소실/폭주/불안정해지는지 같이 확인한다.

#### 좋은 결과 기준

좋은 결과는 단순히 RMSE가 낮은 것이 아니다. 다음 조건을 함께 만족해야 한다.

- validation loss가 train loss와 함께 안정적으로 감소한다.
- gradient norm이 초반부터 0에 붙거나 과도하게 튀지 않는다.
- 예측 수익률이 0 근처에만 몰리지 않는다.
- 예측 수익률 분산이 실제 수익률 분산 대비 지나치게 작지 않다.
- KRW 복원 예측이 단순 persistence baseline과 사실상 동일하지 않다.
- 방향성 지표가 우연 수준에만 머물지 않는다.

#### 나쁜 결과 기준

다음 중 하나라도 반복되면 shortcut collapse 위험이 높다.

- loss는 감소하지만 예측 수익률 표준편차가 거의 0이다.
- loss는 감소하지만 gradient norm이 매우 작아져 사실상 업데이트가 죽는다.
- gradient norm이 특정 epoch에서 과도하게 튀고 이후 validation loss가 회복되지 않는다.
- `near_zero_return_share`가 0.70 이상이다.
- `variance_ratio`가 0.10 미만이다.
- `copy_risk_ratio`가 1에 가깝고 persistence baseline과 차이가 없다.
- 예측 KRW 가격선이 직전가 baseline과 거의 겹친다.

#### 설계 근거 문서

전처리, 정규화, 손실함수, optimizer, gradient policy를 왜 넣었는지는 이 보고서의 아래 "8번 확장 학습 case 설계 근거" 절에 정리돼 있다(2026-10-14 이전에는 별도 문서였다).

> 원문: 구 `test/research_materials/optimization_training_case_design_20260616.md` — 8번 학습 case 설계 근거

### 8번 확장 학습 case 설계 근거

#### 핵심 메시지

이번 문제는 “LSTM 대신 Transformer를 쓸 것인가” 같은 단순 알고리즘 선택 문제가 아니다. 같은 기본 모델 5개(`Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`)도 전처리, 정규화, 손실함수, optimizer, scheduler, gradient clipping, 앙상블 방식에 따라 loss landscape를 이동하는 방식이 달라진다. 따라서 8번은 모델 이름 확장이 아니라 **학습 경로 확장 실험**이어야 한다.

#### 참고한 연구 흐름

##### 1. 비정상 시계열 정규화

- Non-stationary Transformer는 실제 시계열의 분포가 시간에 따라 바뀌면 Transformer 성능이 크게 흔들릴 수 있고, 단순 stationarization이 오히려 중요한 비정상 정보를 지울 수 있다고 본다. 그래서 series stationarization과 de-stationary attention을 같이 쓴다.
- Dish-TS는 input window와 output horizon 사이에도 분포 shift가 생긴다고 보고, input-space와 output-space의 분포를 따로 추정하는 구조를 제안한다.
- FAN은 RevIN류가 평균/분산 같은 기본 trend는 다루지만 seasonal/frequency 성분에는 약하다고 보고, frequency component를 정규화 축으로 넣는다.
- NoRIN은 RevIN류 affine normalization이 heavy-tail과 skewness를 바꾸지 못한다고 지적하고, arcsinh 계열의 비선형 reversible normalization을 제안한다.

8번 반영:

- `window_standard`, `window_robust`, `window_minmax`
- `revin`, `robust_revin`, `asinh_revin`
- `asinh`, `ema_residual`, `frequency_highpass`

##### 2. 단순 모델과 분해/패치 구조의 중요성

- DLinear/NLinear 계열은 복잡한 Transformer보다 단순 linear decomposition이 긴 시계열 예측에서 강할 수 있음을 보인다.
- PatchTST는 긴 sequence를 그대로 attention에 넣기보다 patch 단위 토큰화와 channel-independence가 계산량과 표현력을 동시에 개선할 수 있다고 본다.

8번 반영:

- `DLinearLike`, `NLinearLike`, `PatchTSTLike`
- 기본 5개 모델도 같은 preprocessing/normalization/loss/optimizer 조합으로 다시 확인

##### 3. 손실함수와 평가 목표

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

##### 4. gradient와 optimizer 안정화

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

#### 8번에서 봐야 할 질문

- 같은 `LSTM`이라도 `window_standard + return_huber`와 `asinh_revin + anti_collapse_huber`의 예측 분산이 다른가?
- 같은 `Transformer`라도 cosine scheduler와 plateau scheduler가 0 수익률 붕괴 정도를 다르게 만드는가?
- robust/asinh 계열 전처리가 extreme candle 이후 loss 폭주를 줄이는가?
- frequency/highpass 전처리가 직전가 복사형 baseline과의 중첩을 줄이는가?
- gradient norm이 초반부터 거의 0으로 죽거나 반대로 튀는 case가 특정 모델/손실/정규화에 몰리는가?
- 앙상블은 단순히 MAE만 낮추는가, 아니면 collapse 지표도 같이 낮추는가?

#### 주요 참고 링크

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

---

## 기초통계량(describe 표)

> 2026-10-14 소급 추가(AGENTS.md 2.9k). 실행 당시가 아니라 현재 `data/upbit_data.db`에서 2024-05-28 23:45 ~ 2024-07-11 12:00(끝 미포함)을 다시 조회해 계산했다(업비트 과거 캔들은 바뀌지 않으므로 같은 창이면 같은 값이다). 데이터 범위 근거: 8번은 당시 `btc_15m_advance`의 앞 40,000행(2023-05-20 18:30 ~ 2024-07-11 11:30)을 읽었지만 학습·평가 창은 뒤쪽 4,096개(stride 1, 입력 64봉)만 썼다. 현재 DB는 KRW-BTC가 2023-07-19 19:00부터라 읽은 범위 전체는 재현할 수 없으므로, 실제 학습·평가에 들어간 구간(약 4,161행)을 싣는다. 이 구간의 시작 시각은 현재 DB의 행 순서로 역산한 근사값이다. 당시 읽은 범위의 종가는 32,578,000 ~ 104,937,000 KRW였다(본문).

#### describe 표(pandas `describe()` 그대로, 종목별)

> 숫자 형식: count는 정수, 나머지는 소수 둘째 자리(1 미만 값은 유효숫자 4자리). 로그수익률은 15분봉 기준 % 단위다. 행 단위 원자료라 무체결 구간(캔들 없음)은 포함되지 않는다.

**종가(KRW)**

| 종목 | count | mean | std | min | 25% | 50% | 75% | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KRW-BTC | 4,161 | 90,751,165.83 | 5,699,314.38 | 77,464,000.00 | 86,199,000.00 | 91,981,000.00 | 95,085,000.00 | 99,620,000.00 |

**거래량(코인 수)**

| 종목 | count | mean | std | min | 25% | 50% | 75% | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KRW-BTC | 4,161 | 26.97 | 39.46 | 0.0008536 | 9.13 | 16.08 | 29.32 | 773.38 |

**거래대금(KRW)**

| 종목 | count | mean | std | min | 25% | 50% | 75% | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KRW-BTC | 4,161 | 2,401,942,138.85 | 3,366,948,295.83 | 73,895.40 | 841,136,113.06 | 1,454,084,846.87 | 2,628,429,838.65 | 60,324,802,341.36 |

**로그수익률(%)**

| 종목 | count | mean | std | min | 25% | 50% | 75% | max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KRW-BTC | 4,160 | -0.00338 | 0.2076 | -1.96 | -0.09254 | -0.002121 | 0.09011 | 1.77 |


---


## 1. 연구 목적

8번 실험은 6번에서 정리한 학습 안정화 기준과 7번의 실행 계획을 실제 GPU 학습으로 연결한 첫 번째 breadth 비교다. 목적은 최신 모델의 단순 성능 순위를 정하는 것이 아니라, 업비트 비정상 시계열에서 다음 문제가 모델 계열을 바꾸어도 반복되는지 확인하는 것이다.

- 예측 로그수익률이 0 근처로 몰리는가
- 예측 가격이 직전 가격을 복사하는 persistence 해와 사실상 같아지는가
- 반대로 예측 분산이 실제보다 지나치게 커져 폭주하는가
- 낮아진 학습 손실이 실제 테스트 일반화로 이어지는가

이번 실행은 `breadth_probe`이므로 전처리·정규화·손실함수 조건을 동일하게 고정하고 모델 구조만 비교했다. 따라서 이번 결과로 전처리 방식의 우열까지 결론 내릴 수는 없다.

## 2. 데이터와 실행 환경

- 데이터: `btc_15m_advance`
- 관측치: 39,935개
- 기간: 2023-05-21 10:30 ~ 2024-07-11 11:30
- 종가 범위: 32,578,000 ~ 104,937,000 KRW
- 15분 로그수익률 표준편차: 0.002240
- 로그수익률 왜도: 0.0144
- 로그수익률 첨도: 22.2043
- 결측 셀: 0
- GPU: NVIDIA GeForce RTX 4090 24GB
- PyTorch: 2.10.0+cu126
- Python: 3.13.13
- 학습 window: train 2,867 / validation 614 / test 615

로그수익률 왜도는 0에 가깝지만 첨도는 정규분포의 기준보다 매우 높다. 이는 상승과 하락의 비대칭보다 극단적 변동이 자주 나타나는 두꺼운 꼬리가 더 중요한 문제라는 뜻이다. 평균과 표준편차만 맞추는 전처리로는 급등락 구간의 영향을 충분히 제어하지 못할 수 있다.

## 3. 공통 실험 조건

모든 모델에 다음 조건을 동일하게 적용했다.

- feature set: `wide_stationary`
- preprocessing: `none`
- normalization: `window_standard`
- target: 다음 시점 로그수익률
- loss: `return_huber`
- optimizer: AdamW
- scheduler: cosine annealing
- gradient policy: clip norm 1.0
- sequence length: 64
- hidden width: 96
- epochs: 12
- batch size: 48

`wide_stationary`에는 1·4·16·64구간 수익률, 실현변동성, 고저가 범위, EMA 괴리율, 거래량·거래대금 z-score, turnover proxy가 포함된다. 원시 가격 수준을 그대로 넣지 않고 변화율과 국소 통계량을 사용해 직전 가격 복사 경로를 줄이려는 구성이다.

## 4. 지표 정의

### 4.1 Persistence MAE

다음 가격을 현재 가격과 같다고 예측하는 단순 기준선이다. 이번 test set의 persistence MAE는 190,608 KRW다. 복잡한 모델이 이 값보다 나쁘다면 해당 모델은 추가 계산비용에도 불구하고 직전가 복사보다 유용하지 않다.

### 4.2 Copy-risk ratio

`모델 MAE / persistence MAE`로 계산한다.

- 1 미만: persistence보다 개선
- 1 부근: 사실상 persistence와 동급
- 1 초과: persistence보다 악화

### 4.3 Variance ratio

`예측 수익률 분산 / 실제 수익률 분산`이다.

- 1 부근: 실제 변동 폭을 대체로 보존
- 0에 가까움: 예측이 평평해지는 collapse
- 1보다 지나치게 큼: 예측 진폭 폭주

### 4.4 Near-zero return share

예측 로그수익률의 절댓값이 `1e-4`보다 작은 비율이다. 값이 높으면 모델이 방향과 크기를 구분하지 못하고 0수익률 근처로 피신한 것이다.

### 4.5 Collapse score

다음 세 조건을 합산한 진단 점수다.

- near-zero share가 0.70 초과
- variance ratio가 0.10 미만
- copy-risk ratio가 0.95 초과

낮을수록 낫지만, 현재 정의에서는 persistence보다 나쁘기만 해도 1점이 붙는다. 따라서 `collapse_score=1`은 완전한 성공이 아니라 최소 한 가지 위험이 남았다는 뜻이다.

## 5. 결과

| 모델 | MAE(KRW) | Persistence 대비 | Copy risk | Variance ratio | Near-zero share | DA | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| TimesNetLike | 190,972 | +364 | 1.0019 | 0.0025 | 0.4634 | 0.4846 | 2 |
| NLinearLike | 191,023 | +415 | 1.0022 | 0.0044 | 0.3431 | 0.5041 | 2 |
| DLinearLike | 191,971 | +1,363 | 1.0071 | 0.0033 | 0.2423 | 0.4764 | 2 |
| TCN | 192,063 | +1,455 | 1.0076 | 0.0099 | 0.2358 | 0.4862 | 2 |
| ModernTCNLike | 194,524 | +3,916 | 1.0205 | 0.0311 | 0.1463 | 0.4943 | 2 |
| Transformer | 200,679 | +10,071 | 1.0528 | 0.0307 | 0.0553 | 0.5041 | 2 |
| TimeXerLike | 202,613 | +12,005 | 1.0630 | 0.0954 | 0.0911 | 0.4748 | 2 |
| PatchTSTLike | 223,698 | +33,090 | 1.1736 | 0.2714 | 0.0488 | 0.5415 | 1 |
| Linear | 246,196 | +55,588 | 1.2916 | 0.6004 | 0.0276 | 0.5545 | 1 |
| ITransformerLike | 246,578 | +55,970 | 1.2936 | 0.4417 | 0.0455 | 0.4976 | 1 |
| MambaLike | 319,124 | +128,516 | 1.6742 | 1.4836 | 0.0325 | 0.4829 | 1 |
| LSTM | 368,554 | +177,946 | 1.9336 | 2.2848 | 0.0195 | 0.5171 | 1 |
| GRU | 464,220 | +273,612 | 2.4355 | 4.3904 | 0.0179 | 0.5203 | 1 |
| AutoformerLike | 1,635,845 | +1,445,236 | 8.5822 | 59.3942 | 0.0016 | 0.5089 | 1 |

## 6. 결과 해석

### 6.1 모든 모델이 persistence를 이기지 못했다

14개 모델 모두 copy-risk ratio가 1보다 컸다. 가장 낮은 MAE를 기록한 TimesNetLike도 persistence보다 약 364 KRW 나빴다. 차이가 작아 보일 수 있지만, 예측 분산 비율이 0.0025에 불과하므로 실제 패턴을 학습해 persistence에 근접한 것이 아니다. 거의 움직이지 않는 예측으로 persistence와 비슷한 MAE를 얻은 것이다.

따라서 이번 실험의 결론은 “TimesNetLike가 가장 좋다”가 아니다. 정확한 결론은 “TimesNetLike, NLinearLike, DLinearLike, TCN은 0수익률 collapse를 통해 persistence에 근접했다”이다.

### 6.2 낮은 MAE와 정보 있는 예측은 서로 달랐다

TimesNetLike의 방향 정확도는 48.5%이고 variance ratio는 0.0025다. 반면 Linear는 MAE가 더 크지만 방향 정확도는 55.4%, variance ratio는 0.6004다. Linear는 가격 오차 면에서는 나빴지만 실제 수익률 변화 폭과 방향 정보를 더 많이 남겼다.

이 차이는 후속 실험에서 단순 MAE 순위만 사용하면 안 되는 이유다. 모델 선택은 최소한 다음을 함께 보아야 한다.

- persistence 대비 MAE
- 방향 정확도
- 예측 분산 보존
- 0수익률 쏠림
- seed 또는 시간 구간 변화에 대한 안정성

### 6.3 순환신경망은 소실보다 과도한 진폭 문제가 더 컸다

LSTM과 GRU의 variance ratio는 각각 2.28과 4.39다. 예측이 평평해진 것이 아니라 실제보다 크게 흔들렸다. 학습이 나빴다는 사실을 모두 “기울기 소실”로 설명하면 안 된다. 이번 결과에서는 출력 분산 과대, 학습 목적과 KRW 평가의 불일치, 입력 heavy-tail 영향이 더 직접적인 증거다.

### 6.4 AutoformerLike의 과거 우위는 재현되지 않았다

AutoformerLike는 MAE 163만 KRW, variance ratio 59.39로 가장 심한 폭주를 보였다. 3번 실험에서 Autoformer가 좋아 보였던 결과를 구조 자체의 우월성으로 해석할 수 없다는 점이 다시 확인됐다. 현재 target·feature·평가 구조에서는 오히려 가장 불안정했다.

### 6.5 PatchTSTLike와 Linear는 후속 전처리 연구의 기준 모델로 적합하다

PatchTSTLike는 방향 정확도 54.1%, variance ratio 0.271로 완전한 0수익률 collapse보다 덜 평평했다. Linear는 방향 정확도 55.4%, variance ratio 0.600으로 정보 보존 측면에서 가장 해석하기 쉬웠다.

따라서 9번 전처리 연구에서는 다음 세 유형을 함께 유지하는 것이 적절하다.

- Linear: 단순하고 해석 가능한 기준선
- PatchTSTLike: 상대적으로 안정적인 deep backbone
- TimesNetLike 또는 NLinearLike: MAE는 낮지만 collapse가 강한 대조군
- AutoformerLike 또는 GRU: 분산 폭주를 확인하는 실패 대조군

## 7. 그래프가 저장되지 않은 문제

현재 노트북에는 학습 로그와 Markdown 결과표만 있고 `image/png` 출력이 남아 있지 않다. 코드에는 `plt.show()`가 있었지만 실행 당시 inline backend가 실제 이미지 MIME 출력을 노트북에 기록하지 못했다.

9번에서는 실행 시작 시 Jupyter inline backend를 명시적으로 설정하고, 각 단계마다 다음 그림을 노트북 셀에 출력하도록 한다.

- 원시 가격·수익률·변동성
- train/validation/test 시간 분할
- 전처리 전후 시계열
- 전처리 전후 분포·QQ·주파수 성분
- window 예시
- epoch별 train/validation loss·gradient norm·learning rate
- 실제/예측 수익률과 KRW 복원 가격
- 예측구간과 구간 폭
- coverage와 interval width
- 모델 크기·파라미터 수 대비 test error
- preprocessing × model 성능 heatmap

## 8. 두 영상의 적용 가능성

### 8.1 Black–Scholes 영상

[수학자들이 얼마나 돈을 벌고 싶은지 감도 안옴](https://www.youtube.com/watch?v=99BHnu64pu8)은 Black–Scholes와 옵션 가격화의 역사·아이디어를 다룬다. 이 영상이 현재 프로젝트에 주는 핵심은 특정 가격 공식을 그대로 적용하는 것이 아니라, 미래를 하나의 점으로 단정하지 않고 확률분포와 변동성으로 표현해야 한다는 관점이다.

9번에서는 이를 다음처럼 반영한다.

- point forecast 외에 quantile interval을 계산한다.
- 여러 seed 모델의 예측 분산으로 epistemic uncertainty를 본다.
- 구간 coverage와 width를 함께 평가한다.
- 변동성이 큰 구간에서 구간 폭이 실제로 넓어지는지 본다.
- conformal calibration으로 명목 coverage와 실제 coverage 차이를 측정한다.

Black–Scholes의 로그정규분포 가정을 코인 수익률에 그대로 강제하지는 않는다. 이번 데이터의 첨도가 22를 넘기 때문에, 정규성 가정은 오히려 위험을 과소평가할 수 있다.

### 8.2 Double Descent 영상

["교수님, 전공책이 틀렸습니다" | 더블디센트](https://www.youtube.com/watch?v=5ruiphjlOwo)는 모델 복잡도가 증가할 때 test error가 단순 U자가 아니라 interpolation threshold 부근에서 상승한 뒤 다시 감소할 수 있다는 현상을 설명한다.

9번에서는 다음 실험으로 번역한다.

- 같은 backbone에서 hidden width를 단계적으로 늘린다.
- 파라미터 수, train loss, validation loss, test MAE를 함께 그린다.
- seed를 바꾸어 예측 평균과 분산을 분리한다.
- epoch별 validation/test proxy 변화를 기록한다.
- 파라미터 수만이 아니라 데이터 window 수와 학습 epoch도 complexity axis로 본다.

이는 “큰 모델을 쓰면 해결된다”는 의미가 아니다. interpolation threshold 근처의 불안정성을 직접 측정하고, optimizer와 normalization이 어떤 해를 선택하는지 확인하라는 경고로 적용해야 한다.

## 9. 9번 실험으로의 연결

8번에서 모든 모델이 persistence를 넘지 못했고 collapse 형태가 평탄화와 폭주 두 방향으로 갈렸다. 따라서 9번은 모델 수를 더 늘리는 실험이 아니라 다음 질문에 집중한다.

1. heavy-tail과 극단값을 어떤 변환이 가장 안정적으로 완화하는가
2. 추세·저주파·변동성 군집을 분리하면 collapse가 줄어드는가
3. 두 개 이상의 전처리를 조합했을 때 정보가 제거되는지 보존되는지
4. point error가 비슷할 때 어떤 조합이 더 잘 보정된 예측구간을 만드는가
5. 모델 폭·seed·epoch 변화에서 일반화 오차가 얼마나 흔들리는가

## 10. 참고 자료

- [Black–Scholes 관련 영상](https://www.youtube.com/watch?v=99BHnu64pu8)
- [Double Descent 관련 영상](https://www.youtube.com/watch?v=5ruiphjlOwo)
- [공유 대화: 딥러닝 기울기 소실 분석](https://chatgpt.com/share/6a395764-1d2c-83ee-92ac-20ada70cf9db)
- [Belkin et al., Reconciling modern machine-learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)
- [Nakkiran et al., Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [Kim et al., Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p)
- [Fan et al., Dish-TS](https://arxiv.org/abs/2302.14829)
- [Ye et al., Frequency Adaptive Normalization](https://arxiv.org/abs/2409.20371)
- [Piao et al., FredNormer](https://arxiv.org/abs/2410.01860)
- [Jensen et al., Ensemble Conformalized Quantile Regression](https://arxiv.org/abs/2202.08756)
- [Prinzhorn et al., Conformal time series decomposition](https://arxiv.org/abs/2406.16766)

## 11. 결론

8번은 최신 구조를 늘려도 현재 공통 전처리 조건에서는 문제가 해결되지 않는다는 것을 보여줬다. 최저 MAE 모델들은 실제 변동을 거의 제거해 persistence에 접근했고, LSTM·GRU·AutoformerLike는 반대로 출력 분산이 과대해졌다. 따라서 9번은 전처리 조합, 불확실성 보정, 모델 용량과 seed 분산을 동시에 기록하는 진단 실험이어야 한다.
