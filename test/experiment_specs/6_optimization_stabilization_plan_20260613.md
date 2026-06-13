# 6. 최적화 안정화 후속 실험 계획서

작성일: 2026-06-13

## 1. 목적

5번 실험은 성능 리더보드가 아니라 진단 실험이었다. 대표 알고리즘 5개(`Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`)와 대표 목표/손실 조합을 작게 돌려, 비정상 업비트 시계열에서 loss가 줄어도 예측이 `직전가 복사`, `0 수익률`, `평평한 예측`으로 무너질 수 있음을 확인하는 단계였다.

6번 실험은 이 문제를 해결하기 위한 후속 안정화 단계다. 바로 독립변수나 데이터마트를 늘리지 않고, target, normalization, loss, resource scale, model selection gate를 하나씩 바꾼다. 이유는 단순하다. 최적화 문제가 남아 있으면 새로운 변수를 붙여도 성능 변화가 입력 정보 때문인지, 손실함수 때문인지, 우연한 split 때문인지 구분하기 어렵다.

## 2. 5번 코드를 왜 수정했는가

5번은 처음에 “어떤 모델이 잘 맞는가”처럼 읽힐 위험이 있었다. 그러나 현재 연구의 핵심 문제는 더 앞단에 있다. 비정상 가격 시계열에서는 모델이 다음 가격을 진짜로 예측하지 않고도 loss를 줄일 수 있다. 예를 들면 직전 가격을 거의 복사하거나, 다음 수익률을 거의 0으로 예측하면 학습 loss는 낮아 보일 수 있다.

그래서 5번 코드는 다음 목적에 맞게 수정했다.

- 대표 알고리즘 5개를 모두 유지한다.
- `level_mse`, `return_huber`, `return_directional_hybrid`처럼 target과 loss를 명확히 분리한다.
- 결과를 단순 loss가 아니라 `persistence_gap`, `collapse_score`, `variance_ratio`, `zero_share`, `sign_agreement`로 함께 본다.
- 노트북 출력과 이미지 추출 흐름을 보고서 source of truth로 둔다.
- 6번에서 이어 쓸 수 있도록 `normalization`과 `stabilization_loss_probe` 축을 추가한다.

즉 5번은 최종 모델 선택용이 아니라, “어떤 조합이 쉬운 해로 무너지는지”를 빠르게 찾는 기준선이다. 6번은 그 기준선에서 발견한 문제를 고치는 단계다.

## 3. 이번 문제를 더 정확히 정의

문제는 단순히 “기울기 소실”이 아니다. 기울기 소실은 역전파 과정에서 gradient가 너무 작아져 앞쪽 layer가 학습되지 않는 현상이다. 이번 문제는 그보다 “목표 함수가 허용하는 쉬운 해”에 가깝다.

현재 상황을 쉽게 풀면 다음과 같다.

- 가격 레벨은 자기상관이 강하다. 다음 가격은 직전 가격과 매우 비슷한 경우가 많다.
- 그래서 `next close`를 직접 맞히면 모델은 시장 구조를 배우기보다 “직전가를 거의 복사하는 답”을 찾기 쉽다.
- 로그수익률 target으로 바꾸면 복사형 문제는 줄지만, 이번에는 변동이 작은 구간이 많아 “0 수익률 근처로 납작하게 예측하는 답”이 생길 수 있다.
- loss가 줄어도 persistence baseline보다 못하면 예측 모델로는 실패다.

따라서 6번의 통과 기준은 “loss가 내려갔다”가 아니다. 최소 기준은 다음과 같다.

- `persistence_gap < 0`: 모델 MAE가 직전가 복사 baseline보다 낮아야 한다.
- `variance_ratio`가 0에 붙지 않아야 한다. 예측 변화폭이 실제 변화폭 대비 너무 작으면 flat prediction이다.
- `zero_share`가 과도하게 높지 않아야 한다. 예측 수익률이 0 근처로 몰리면 0-return shortcut이다.
- `sign_agreement`가 무작위 수준인 0.5 부근에서 벗어나는지 확인한다.
- 같은 결론이 작은 window에서만이 아니라 2048 window 수준에서도 유지되는지 본다.

## 4. 참고 문헌과 적용 아이디어

### 4.1 비정상성 완화

Non-stationary Transformers는 실제 시계열의 평균과 분산이 계속 바뀌는 문제가 예측을 어렵게 만든다고 보고, stationarization과 de-stationary attention을 함께 제안한다. 이번 실험에서는 Transformer를 그대로 키우기보다 먼저 입력 스케일과 목표 정의를 안정화해야 한다는 근거로 사용한다.

- 참고: Liu et al., 2022, Non-stationary Transformers, https://arxiv.org/abs/2205.14415
- 적용: `standard`, `robust`, `window_standard` normalization 비교
- 확인: gradient norm, validation-train gap, persistence gap

### 4.2 분포 이동 대응 정규화

RevIN은 시계열의 평균/분산 shift를 instance 단위 normalization과 역복원으로 다룬다. 5번/6번에서는 완전한 RevIN layer를 새로 넣기 전에, 각 입력 window 내부 값만 사용해 표준화하는 `window_standard`를 경량 대안으로 먼저 둔다.

- 참고: Kim et al., 2021, RevIN, https://openreview.net/forum?id=cGDAkQo1C0p
- 적용: `window_standard`
- 해석: validation gap이 줄고 persistence gap이 나아지면 입력 분포 이동이 실제 원인 후보가 된다.

### 4.3 손실함수 재설계

MSE는 평균 예측에 끌리기 쉽다. 금융 시계열처럼 작은 변동이 많고 큰 변동이 드물게 나오는 데이터에서는 “평균적으로 덜 틀리는 예측”이 실전적으로는 쓸모없는 예측일 수 있다.

- 적용: `return_huber`, `return_directional_hybrid`, `return_vol_weighted_mse`, `return_tail_focus`
- 해석: 방향성만 좋아지고 KRW MAE 또는 persistence gap이 나빠지면 실패다.
- 기준: 방향성, baseline 돌파, collapse score를 함께 봐야 한다.

### 4.4 모델 복잡도와 일반화

SAM은 training loss 자체만 낮추는 것보다 sharpness를 함께 고려해야 일반화가 좋아질 수 있다고 본다. 이번 6번에서는 SAM까지 바로 넣지는 않는다. 대신 validation gap, gradient norm, persistence gap을 함께 보고 “loss 최저 epoch가 정말 좋은 epoch인가”를 먼저 분리한다.

- 참고: Foret et al., 2020, Sharpness-Aware Minimization, https://arxiv.org/abs/2010.01412
- 적용 후보: 다음 pass에서 collapse-aware early stopping 또는 SAM/ASAM 비교
- 현재 적용: loss 최저가 아니라 baseline-relative 지표를 함께 보는 selection gate

### 4.5 설명 가능한 다변량 확장

Temporal Fusion Transformer는 static/known/observed covariate와 variable selection을 결합해 다변량 시계열을 해석 가능하게 다룬다. 다만 지금 단계에서는 TFT류 구조를 먼저 넣지 않는다. 최적화 붕괴가 해결된 뒤에 독립변수와 데이터마트를 붙여야 설명이 가능하기 때문이다.

- 참고: Lim et al., 2019, Temporal Fusion Transformers, https://arxiv.org/abs/1912.09363
- 적용 후보: 7번 이후 text/context/historical-flow 변수를 붙일 때 variable selection 또는 gating
- 현재 보류 이유: 지금은 입력 변수를 늘리는 것보다 target/loss/normalization 안정화가 먼저다.

## 5. 실행 자원 원칙

사용자 경험상 이전 3번 ipynb에서는 2048 수준으로 15개 알고리즘 학습이 가능했지만, 전체 수행 시간이 약 5시간까지 걸렸다. 이번에는 오래 걸려도 되지만 OOM으로 중간에 죽으면 해석이 끊기므로, 다음 원칙을 둔다.

- GPU 학습 stage는 동시에 병렬 실행하지 않는다.
- 병렬 처리는 CSV/이미지/보고서 후처리처럼 메모리 위험이 작은 작업에만 적용한다.
- 기본 `batch_size=64`, 최소 `min_batch_size=16`으로 둔다.
- CUDA OOM이 감지되면 같은 명령을 batch size 절반으로 자동 재시도한다.
- `max_windows=2048`은 본 실행 후보로 두되, 커널 상태가 불안정하면 `server_1024` 또는 `server_light` 프로필로 낮춘다.
- `num_workers=0`이 기본이다. 서버 I/O가 병목일 때만 조절한다.

## 6. 실행 명령

계획만 확인:

```bash
uv run test/models/6_optimization_stabilization_test.py
```

서버에서 전체 stage를 순차 실행:

```bash
uv run test/models/6_optimization_stabilization_test.py --run-all --profile server_2048
```

OOM 걱정이 있거나 커널 상태를 먼저 확인할 때:

```bash
uv run test/models/6_optimization_stabilization_test.py --run-all --profile server_1024
```

특정 stage만 실행:

```bash
uv run test/models/6_optimization_stabilization_test.py --run-stage 2 --profile server_2048
```

명령만 출력하고 실행하지 않기:

```bash
uv run test/models/6_optimization_stabilization_test.py --run-all --profile server_2048 --dry-run
```

## 7. Stage별 설계

### Stage 0. 5번 quick_probe 재현

- 목적: 현재 5번 결과가 승인된 서버 환경에서도 재현되는지 확인
- 바꾸는 것: 없음
- 좋은 그림: 15개 케이스의 loss curve, persistence gap, collapse score 방향이 기존 5번 보고서와 크게 다르지 않다.
- 나쁜 그림: seed나 split이 같은데 순위와 곡선 방향이 완전히 달라진다.
- 해석: 재현이 안 되면 이후 실험은 비교가 아니라 다른 실험이 된다. 이 경우 데이터 기간, split, scaler, seed부터 고정한다.

### Stage 1. Target/objective gate

- 목적: `level_mse`를 기본 후보에서 제외해도 되는지 return target 중심으로 다시 확인
- 바꾸는 것: target/objective만 확장
- 좋은 그림: return 계열에서 validation loss가 같이 내려가고 persistence gap이 0 아래로 내려가는 후보가 생긴다.
- 나쁜 그림: return target도 variance ratio가 0 근처이고 persistence gap이 계속 양수다.
- 해석: return target도 실패하면 다음 target을 volatility-scaled return, neutral-band direction label, triple-barrier style label로 바꿔야 한다.

### Stage 2. Normalization ablation

- 목적: 비정상성 완화가 실제로 gradient와 validation gap을 안정화하는지 확인
- 바꾸는 것: `robust`, `window_standard`, `identity`
- 좋은 그림: gradient norm 튐이 줄고 validation-train gap과 persistence gap이 함께 낮아진다.
- 나쁜 그림: 정규화만 바꿨는데 zero_share가 더 높아지고 sign agreement가 무작위 수준이다.
- 해석: `identity`가 좋아 보이면 기존 scaler가 문제였을 가능성이 있고, `window_standard`가 좋아지면 구간별 평균/분산 shift가 주요 원인 후보가 된다.

### Stage 3. Loss ablation

- 목적: Huber, directional hybrid, volatility-weighted, tail-focus loss 중 붕괴를 덜 만드는 후보 선정
- 바꾸는 것: loss/objective만 변경
- 좋은 그림: sign agreement가 올라가면서 persistence gap이 0 아래로 내려가고 variance ratio가 0에 붙지 않는다.
- 나쁜 그림: 방향성 지표만 좋아지고 KRW MAE 또는 persistence gap이 나빠진다.
- 해석: 금융 예측에서는 방향만 맞히는 것과 baseline을 이기는 것은 다르다. 둘 중 하나만 좋으면 아직 채택하지 않는다.

### Stage 4. Resource scale check

- 목적: `max_windows=2048`, 더 긴 epoch에서도 같은 결론이 유지되는지 확인
- 바꾸는 것: 창 수와 epoch를 본 실행 후보 수준으로 확장
- 좋은 그림: stage 3에서 좋았던 조합이 window를 키워도 같은 방향으로 유지된다.
- 나쁜 그림: 작은 창에서는 좋아 보였는데 2048에서 persistence gap이 다시 양수로 돌아온다.
- 해석: 확장 시 무너지면 모델 복잡도, seq_len, normalization부터 다시 줄여 stage 2로 돌아간다.

### Stage 5. Independent variable gate

- 목적: 텍스트 독립변수, historical flow mart, 온체인/유동성 변수 연결 전 최소 안정성 기준 확정
- 바꾸는 것: 이 stage 자체는 문서/판정 gate다.
- 좋은 그림: 최소 한 조합이 baseline을 지속적으로 이기고 flat/zero-return 붕괴 신호가 완화된다.
- 나쁜 그림: 어떤 조합도 baseline을 못 이기거나, 작은 조정마다 결론이 계속 뒤집힌다.
- 해석: 통과 전에는 독립변수 확장보다 target/loss/normalization 재설계를 우선한다.

## 8. 산출물

서버에서 전체 실행 후 다음 파일군을 확인한다.

- `test/results/5_optimization_diagnostics_*_summary.csv`
- `test/results/5_optimization_diagnostics_*_curves.csv`
- `test/results/5_optimization_diagnostics_*.md`
- `test/images/5_optimization_diagnostics_*_training.png`
- `test/images/5_optimization_diagnostics_*_collapse.png`
- `test/images/5_optimization_diagnostics_*_case_*.png`

보고서는 실행된 노트북 또는 저장된 CSV/이미지 기준으로 작성한다. 과거 git 상태나 예전 보고서 수치로 대체하지 않는다.

## 9. 다음 코드 위치

- 실행 계획 코드: `test/models/6_optimization_stabilization_test.ipynb`
- diff 추적 미러: `test/models/6_optimization_stabilization_test.py`
- 재사용되는 진단 엔진: `test/models/5_optimization_diagnostics_test.ipynb`
- diff 추적 미러: `test/models/5_optimization_diagnostics_test.py`

## 10. 보고서 작성 원칙

- 각 stage마다 `무엇을 시험했는지`, `왜 이 단계가 필요한지`, `좋은 그림이면 어떻게 보여야 하는지`, `이번 그림은 왜 그렇거나 그렇지 않은지`, `다음 수정 여부`를 기록한다.
- 결과 해석은 현재 로컬 또는 서버에서 실제 실행된 `.ipynb` 출력과 추출 이미지 기준으로만 작성한다.
- 독립변수/데이터마트 확장은 6번 gate를 통과한 뒤 진행한다.
