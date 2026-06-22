# 10번 Objective·Ensemble 본실험 계획

## 1. 왜 10번을 별도 실험으로 만드는가

8번에서는 14개 모델 모두 단순 직전가 유지 방식인 persistence MAE를 이기지 못했다. 낮은 MAE를 보인 일부 모델은 실제 변동을 학습한 것이 아니라 예측 수익률을 0 근처로 축소했고, LSTM·GRU·Autoformer-like 계열은 반대로 실제보다 큰 출력 분산을 만들었다.

9번은 이 문제를 전처리 관점에서 분리했다. 실제 완료된 실행은 28개 전처리, 4개 모델, seed 42, hidden 96의 preprocessing matrix 112개다. 가장 낮은 MAE는 `PatchTSTLike + seasonal_diff16`의 약 227,412원이었지만 persistence 약 190,608원보다 19.3% 컸다. `PatchTSTLike + winsor_025`는 방향 정확도 약 55.45%였지만 MAE는 persistence보다 26.3% 컸다. TimesNetLike는 0수익률 평탄화, AutoformerLike는 출력 분산 폭주를 보였다.

10번은 9번을 수정하지 않고 새 번호로 분리한다. 전처리 후보는 입력 조건으로 고정하고, 모델이 무엇을 최소화하도록 요구할지와 여러 seed·모델 예측을 어떻게 결합할지를 본실험으로 확인한다.

## 2. 핵심 연구 질문

1. Huber 오차만 최소화하는 것보다 방향, 분산, 상관, tail, regime 항을 함께 사용하면 persistence 미달이 줄어드는가?
2. 0수익률 평탄화를 막는 항이 출력 분산 폭주라는 반대 문제를 만들지는 않는가?
3. validation 결과만으로 선택한 seed/model ensemble이 단일 모델보다 안정적인가?
4. 단일 seed에서만 좋아 보이는 결과가 다른 seed에서도 재현되는가?

## 3. 고정 조건

- 데이터: DuckDB `btc_15m_advance`
- target: 다음 시점 로그수익률
- 원본 스케일 평가: `prev_close * exp(pred_return)`으로 KRW 가격 복원
- feature set: `wide_stationary`
- normalization: `window_standard`
- optimizer: AdamW
- scheduler: cosine
- gradient policy: clip norm 1.0
- 주 실험 모델: Linear, PatchTSTLike
- 실패 통제 모델: TimesNetLike, AutoformerLike
- 전처리 후보: seasonal difference 16, frequency band-pass, median residual 5, linear detrend+robust asinh, winsor 2.5%, no preprocessing
- 시간 분할: train 70%, validation 15%, test 15%
- test 결과는 모델 선택이나 ensemble 가중치 계산에 사용하지 않는다.

## 4. Objective 정의

### Huber

큰 오차의 제곱 페널티를 완화해 heavy-tail 수익률에서 극단값 하나가 전체 학습을 지배하지 않도록 한다. 다만 단독 사용 시 0수익률 부근의 안전한 답을 선택할 수 있다.

### Directional Huber

Huber에 실제 수익률 부호와 반대 방향을 예측했을 때의 softplus 페널티를 추가한다. 값 오차가 작더라도 방향을 계속 틀리는 모델을 억제한다.

### Variance Huber

예측 표준편차와 실제 표준편차 비율의 로그 절댓값을 Huber에 추가한다. 예측이 0으로 평평해지는 문제와 지나치게 폭주하는 문제를 대칭적으로 벌점화한다.

### Correlation Huber

배치 안에서 예측과 실제 수익률의 상관이 낮을 때 페널티를 준다. 평균값만 맞히는 대신 상대적인 상승·하락 순서를 학습하도록 유도한다.

### Tail Huber

실제 수익률 절댓값이 큰 관측치에 더 큰 가중치를 준다. 일반 구간만 잘 맞히고 급등락 구간을 무시하는 문제를 확인한다.

### Regime Huber

배치 내부의 상위 25% 변동 구간과 나머지 구간의 손실을 같은 비중으로 평균한다. 관측 수가 많은 저변동 구간이 목적함수를 독점하지 않게 한다.

### Anti-collapse v2

Huber, 방향, 분산, near-zero 페널티를 함께 사용한다. 예측이 0 근처로 몰리는 현상을 직접 억제하되 분산 폭주 여부를 함께 확인한다.

### Balanced Composite

Huber, 방향, 분산, 상관, 평균 편향을 작은 가중치로 함께 사용한다. 단일 보조항이 학습을 지배하지 않도록 구성한 종합 objective다.

보조항은 Huber와 단위 및 크기가 다르다. 예를 들어 방향 손실은 약 0.6 수준인데 Huber는 약 0.001 수준일 수 있다. 단순 가중합을 하면 방향 항이 전체 학습을 지배할 수 있다. 10번에서는 각 보조항의 detached magnitude를 Huber 크기에 맞춘 뒤 상대 가중치를 적용하고, 첫 3 epoch 동안 보조항 비중을 점진적으로 높인다.

## 5. Suite 구성

### objective_screen

- 6개 전처리
- Linear, PatchTSTLike
- 8개 objective
- seed 42
- 총 96개 케이스

목적은 objective별 실패 형태를 넓게 거르는 것이다.

### failure_mode_control

- 모델: TimesNetLike, AutoformerLike
- 전처리: none, seasonal_diff16, winsor_025
- objective: Huber, variance Huber, anti-collapse v2, balanced composite
- 총 24개 케이스

TimesNetLike의 평탄화와 AutoformerLike의 분산 폭주가 objective 변경으로 실제 완화되는지 확인한다. 이 두 모델은 우승 후보가 아니라 실패 유형 통제군이다.

### seed_confirmation

- objective screen에서 사용할 대표 objective 최대 4개
- seed 42, 137, 2026
- 여러 모델과 전처리에서 결과 재현 확인

단일 seed 우승을 연구 결론으로 채택하지 않기 위한 단계다.

### ensemble_confirmation

- 여러 seed·모델·objective 결과를 validation selection score로 정렬
- validation 상위 `top-k`만 선택
- simple mean, validation-weighted mean, median ensemble 비교
- test 결과는 선택 과정에 사용하지 않음

### full

objective, preprocessing, model, seed 전체 조합과 ensemble을 수행한다. 실행 시간이 길기 때문에 objective screen과 seed confirmation 이후에 수행한다.

## 6. 평가 기준

- `copy_risk_ratio = model MAE / persistence MAE`
  - 1 미만이어야 persistence를 이긴다.
- `variance_ratio = Var(predicted return) / Var(actual return)`
  - 1에 가까울수록 실제 변동 폭을 보존한다.
  - 0에 가까우면 평탄화, 지나치게 크면 출력 폭주다.
- `direction_accuracy`
  - 0.5 부근은 동전 던지기 수준이다.
- `near_zero_return_share`
  - 예측이 0수익률 주변에 몰리는 정도다.
- `collapse_score`
  - near-zero, variance, persistence 실패를 결합한 경고값이다.
- `validation_selection_score`
  - validation copy-risk, variance 불일치, 방향 실패, near-zero 쏠림을 합친 ensemble 선택 지표다.

## 7. 좋은 그림과 실패한 그림

좋은 학습 곡선은 train과 validation objective가 함께 감소하고 특정 보조항만 급격히 커지지 않는다. 예측 수익률 그래프는 실제 수익률의 부호와 변동 폭을 일부 따라가며, KRW 복원 그래프는 persistence와 완전히 겹치지 않으면서 실제 가격에 더 가까워야 한다.

실패한 그림은 다음 두 종류다.

- 평탄화: 예측 수익률이 0 주변의 가는 선이 되고 variance ratio가 0에 가까워진다.
- 분산 폭주: 예측 수익률 진폭이 실제보다 훨씬 커지고 KRW 예측이 실제 가격 범위를 벗어난다.

## 8. 서버 실행 순서

### 10번과 11번 병렬 실행 원칙

10번은 다음 15분 수익률을 직접 예측하는 `점예측 가설`이다. 11번은 향후 16개 15분봉, 즉 약 4시간 안의 급변·하방 위험 발생 확률을 예측하는 `위험 이벤트 가설`이다. 두 실험은 서로의 실행 결과를 파일로 읽지 않으므로 동시에 시작할 수 있다.

학교 서버의 RTX 4090 한 장을 공유하므로 10번 기본 `point_primary` slot은 다음처럼 제한한다.

- GPU allocator 상한: 전체 VRAM의 52%
- batch size: 32
- DataLoader workers: 2
- torch threads: 8

11번은 별도 커널에서 `risk_secondary` slot을 사용한다. 두 slot의 GPU allocator 상한 합계는 약 88%이며, CUDA context와 일시적 메모리를 위해 나머지를 비워 둔다. OOM이 발생하면 각 실험의 기존 batch 절반 재시도 로직이 작동한다.

두 노트북을 동시에 실행할 때는 10번 노트북을 첫 번째 venv 커널, 11번 노트북을 두 번째 venv 커널에 연결한다. 두 venv가 GPU를 분리하는 것은 아니므로 `exclusive` slot 두 개를 동시에 사용하면 안 된다.

### 문법과 case 구성 확인

```bash
python test/models/10_objective_ensemble_confirmation_test.py \
  --suite objective_screen \
  --parallel-slot point_primary \
  --max-cases 8 \
  --dry-run
```

### 소형 GPU 확인

```bash
python test/models/10_objective_ensemble_confirmation_test.py \
  --suite parallel_point_probe \
  --parallel-slot point_primary \
  --models Linear,PatchTSTLike \
  --preprocessings seasonal_diff16,winsor_025 \
  --objectives huber,balanced_composite \
  --seeds 42,137,2026 \
  --epochs 20 \
  --max-windows 4096 \
  --max-cases 24 \
  --ensemble-top-k 5 \
  --continue-on-failure
```

이 24개 실험은 10번 점예측 연구선의 종료시험이다. 여러 seed에서 persistence를 이기지 못하면 objective 조합을 계속 늘리지 않고 11번 위험 이벤트 가설과 비교한다.

### 단독 실행이 필요한 경우

```bash
python test/models/10_objective_ensemble_confirmation_test.py \
  --suite objective_screen \
  --parallel-slot exclusive \
  --epochs 20 \
  --max-windows 4096 \
  --continue-on-failure
```

### 평탄화·폭주 통제군

```bash
python test/models/10_objective_ensemble_confirmation_test.py \
  --suite failure_mode_control \
  --epochs 12 \
  --max-windows 4096 \
  --continue-on-failure
```

### Seed 및 ensemble 확인

```bash
python test/models/10_objective_ensemble_confirmation_test.py \
  --suite ensemble_confirmation \
  --models Linear,PatchTSTLike \
  --preprocessings seasonal_diff16,frequency_bandpass,median_residual_5,linear_detrend+asinh_robust,winsor_025 \
  --objectives variance_huber,correlation_huber,anti_collapse_v2,balanced_composite \
  --seeds 42,137,2026 \
  --epochs 20 \
  --max-windows 4096 \
  --max-cases 120 \
  --ensemble-top-k 5 \
  --continue-on-failure
```

## 9. 산출물 원칙

- 결과 원본은 `test/models/10_objective_ensemble_confirmation_test.ipynb`의 inline 출력이다.
- 코드에서는 `plt.show()`만 사용한다.
- 서버에 PNG, CSV, Markdown 결과 파일을 자동 저장하지 않는다.
- 보고서용 그림은 로컬에서 `test/scripts/extract_notebook_images.py`로 노트북 출력에서 추출한다.

## 10. 9번 시각화 문제의 수정

9번 예측 그림의 calibration scatter는 세 패널이 x축을 공유해 실제 수익률 좌표가 시점 인덱스 0~299에 눌렸다. 10번에서는 산점도를 독립 x축으로 만들고 다음을 표시한다.

- x축: 실제 다음 로그수익률
- y축: 예측 다음 로그수익률
- 45도 기준선
- Pearson 상관계수
- 실제·예측 수익률에 맞춘 동일한 x/y 범위

산점도 점이 45도선 주변에 모이고 Pearson 상관이 양수로 커져야 실제 변동 관계를 학습했다고 해석한다.
