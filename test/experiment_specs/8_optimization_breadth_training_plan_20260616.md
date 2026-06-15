# 8번 최적화 breadth training 설계

## 역할

8번은 7번의 stage plan을 실제 학습 실험으로 바꾸는 후속 번호다. 5번은 shortcut collapse 진단 기준선, 6번은 안정화 기준 정리, 7번은 자원 인식형 확장 계획이었다. 8번은 이 계보를 보존하면서 실제 GPU 학습, 시각화, 성능표, collapse 진단을 생성한다.

중요한 점은 8번이 단순히 알고리즘 이름만 늘리는 실험이 아니라는 것이다. 같은 5개 기본 모델(`Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`)도 전처리, 정규화, 손실함수, optimizer, scheduler, gradient clipping, 앙상블 방식에 따라 학습 경로가 달라질 수 있다. 따라서 8번은 **알고리즘 breadth**와 **학습 방식 breadth**를 함께 본다.

## 왜 8번이 필요한가

7번 결과는 모델 성능 결과가 아니라 계획 출력이었다. 따라서 “Autoformer-like가 실제로 개선되는가”, “PatchTST-like가 0 수익률 붕괴를 줄이는가”, “DLinear/NLinear-like 같은 단순 구조가 복잡한 Transformer보다 안정적인가”, “앙상블이 직전가 복사 문제를 줄이는가”를 판단할 수 없었다.

또한 7번은 같은 모델의 학습 방식을 바꾸는 실험을 충분히 포함하지 않았다. 하지만 비정상 업비트 시계열에서는 모델 구조보다 preprocessing, normalization, loss, gradient flow가 더 큰 영향을 줄 수 있다. 그러므로 8번은 모델 family 확장과 함께 학습 조건 확장을 동시에 수행한다.

## 고정할 안정화 기준

- Target: 다음 가격 레벨이 아니라 다음 로그수익률
- Evaluation: 예측 로그수익률을 KRW 가격으로 복원한 MAE/RMSE
- Preprocessing: `none`, `winsorize`, `asinh`, `diff`, `ema_residual`, `frequency_highpass`
- Normalization: `window_standard`, `window_robust`, `window_minmax`, `revin`, `robust_revin`, `asinh_revin`, `global_standard`
- Loss: `return_mse`, `return_mae`, `return_huber`, `return_huber_directional`, `vol_weighted_huber`, `anti_collapse_huber`, `quantile_pinball`
- Optimizer/Scheduler: `adamw`, `adam`, `rmsprop`, `sgd_momentum`와 `cosine`, `onecycle`, `plateau`, `none`
- Gradient policy: `clip0.5`, `clip1`, `clip5`, `adaptive`, `none`
- Model selection: validation loss만 보지 않고 persistence gap, variance ratio, near-zero return share, copy-risk ratio를 함께 확인

## 포함 모델군

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

## 추가 suite

- `breadth_probe`: 기본 안정화 조건에서 모델 family 자체를 넓게 비교한다.
- `core_tuning_probe`: 6번의 5개 기본 모델을 유지하되 전처리, 정규화, 손실함수, optimizer, scheduler, gradient clipping을 함께 흔들어 본다.
- `preprocessing_cross_check`: 모델을 고정하고 `winsorize`, `asinh`, `diff`, `ema_residual`, `frequency_highpass`가 collapse 지표에 미치는 영향을 본다.
- `normalization_cross_check`: `window_standard`, `window_robust`, `window_minmax`, `revin`, `robust_revin`, `asinh_revin`을 비교한다.
- `loss_cross_check`: Huber, 방향성 벌점, 변동성 가중, anti-collapse, pinball loss를 비교한다.
- `optimization_cross_check`: optimizer와 scheduler 조합이 같은 모델의 학습 경로를 어떻게 바꾸는지 본다.
- `gradient_cross_check`: gradient clipping 강도와 adaptive clipping이 loss/gradient norm/붕괴 지표를 어떻게 바꾸는지 본다.
- `ensemble_probe`: 단일 모델 결과를 묶어 앙상블이 collapse와 KRW MAE를 동시에 줄이는지 본다.
- `full`: 넓은 조합을 순차적으로 확인하되, 서버 시간이 길어질 수 있으므로 중간 CSV/PNG 저장을 전제로 한다.

## 서버 자원 기준

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
- 중간 결과는 case마다 CSV/PNG로 저장

## 실행 예시

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

## 산출물

- `test/results/8_optimization_breadth_training_<timestamp>_summary.csv`
- `test/results/8_optimization_breadth_training_<timestamp>_curves.csv`
- `test/results/8_optimization_breadth_training_<timestamp>_report.md`
- `test/images/8_optimization_breadth_training_<timestamp>_*_learning_curve.png`
- `test/images/8_optimization_breadth_training_<timestamp>_*_prediction.png`
- `test/images/8_optimization_breadth_training_<timestamp>_collapse_diagnostics.png`

`curves.csv`에는 `train_loss`, `val_loss`뿐 아니라 `grad_norm_mean`, `grad_norm_max`, `lr`도 저장한다. 따라서 loss만 줄어드는지, gradient가 소실/폭주/불안정해지는지 같이 확인한다.

## 좋은 결과 기준

좋은 결과는 단순히 RMSE가 낮은 것이 아니다. 다음 조건을 함께 만족해야 한다.

- validation loss가 train loss와 함께 안정적으로 감소한다.
- gradient norm이 초반부터 0에 붙거나 과도하게 튀지 않는다.
- 예측 수익률이 0 근처에만 몰리지 않는다.
- 예측 수익률 분산이 실제 수익률 분산 대비 지나치게 작지 않다.
- KRW 복원 예측이 단순 persistence baseline과 사실상 동일하지 않다.
- 방향성 지표가 우연 수준에만 머물지 않는다.

## 나쁜 결과 기준

다음 중 하나라도 반복되면 shortcut collapse 위험이 높다.

- loss는 감소하지만 예측 수익률 표준편차가 거의 0이다.
- loss는 감소하지만 gradient norm이 매우 작아져 사실상 업데이트가 죽는다.
- gradient norm이 특정 epoch에서 과도하게 튀고 이후 validation loss가 회복되지 않는다.
- `near_zero_return_share`가 0.70 이상이다.
- `variance_ratio`가 0.10 미만이다.
- `copy_risk_ratio`가 1에 가깝고 persistence baseline과 차이가 없다.
- 예측 KRW 가격선이 직전가 baseline과 거의 겹친다.

## 설계 근거 문서

전처리, 정규화, 손실함수, optimizer, gradient policy를 왜 넣었는지는 `test/research_materials/optimization_training_case_design_20260616.md`에 별도 정리한다.
