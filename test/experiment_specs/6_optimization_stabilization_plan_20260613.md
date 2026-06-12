# 6. 최적화 안정화 후속 실험 계획서

작성일: 2026-06-13

## 목적

5번 실험은 대표 알고리즘 5개와 목적함수 3개를 작게 돌려, 비정상 업비트 시계열에서 손실이 줄어도 예측이 `직전가 복사`, `0 수익률`, `평평한 예측`으로 무너질 수 있음을 확인한 진단 단계다.

6번 실험은 바로 독립변수를 늘리는 단계가 아니다. 5번에서 드러난 최적화 붕괴를 줄이기 위해 전처리, 데이터 구성, 손실함수, 학습 방식, 모델 선택 기준을 하나씩 바꾸며 무엇이 실제로 문제를 줄이는지 확인하는 안정화 단계다.

## 왜 5번과 6번을 나누는가

- 5번은 `작게 확인하는 진단 단계`다. 대표 구조와 대표 목적함수만으로 어떤 조합이 쉬운 해로 무너지는지 빠르게 확인한다.
- 6번은 `원인별 처방을 검증하는 단계`다. 5번에서 가장 덜 무너진 후보를 기준으로, 한 번에 하나씩 수정해 어떤 처방이 실제로 persistence gap, collapse score, sign agreement를 개선하는지 본다.
- 이 순서를 지키지 않으면 독립변수나 데이터마트를 붙였을 때 성능이 좋아진 이유가 입력 정보 때문인지, 손실함수 재설계 때문인지, 단순 운 때문인지 구분하기 어렵다.

## 5번에서 확인한 문제

- `level_mse`는 5개 대표군 전체에서 가격 레벨 직접 회귀의 한계를 보였다.
- `return_huber`는 손실은 잘 줄었지만, 0 수익률 또는 축소된 분산 예측으로 흐르는 위험이 남았다.
- `directional_hybrid`는 TCN에서 가장 덜 무너졌지만, 개선 폭이 작고 near-zero 예측 비율이 높아 추가 검증이 필요하다.
- Transformer는 표현력이 강하지만 현재 데이터 규모와 목적함수에서는 shortcut을 더 빠르게 찾는 쪽으로 작동할 위험이 있다.

## 문헌과 사례에서 가져올 수 있는 처방

### 1. 비정상성 완화

Non-stationary Transformers는 real-world time series에서 비정상성이 Transformer 예측을 어렵게 만들기 때문에 stationarization과 de-stationary attention을 함께 제안한다. 이 관점은 이번 실험에서 가격 레벨 직접 회귀가 무너진 이유와 연결된다.

- 적용 후보: rolling z-score, robust scaling, regime-wise normalization, RevIN류 instance normalization
- 확인 지표: validation loss가 같이 내려오는지, persistence gap이 기준선 아래로 내려오는지
- 참고: Liu et al., 2022, Non-stationary Transformers, https://arxiv.org/abs/2205.14415

### 2. 분포 이동 대응 정규화

RevIN과 adaptive normalization 계열은 시간에 따라 평균과 분산이 바뀌는 문제를 입력 단위 정규화와 역복원으로 다룬다. 이번 문제에서는 예측을 KRW 원스케일로 복원해야 하므로, 정규화와 역복원을 함께 검증해야 한다.

- 적용 후보: RevIN-like input normalization, rolling median/IQR scaling, train-only scaler 고정
- 확인 지표: KRW MAE, persistence ratio, variance ratio
- 참고: Kim et al., 2021, RevIN, https://openreview.net/forum?id=cGDAkQo1C0p
- 참고: Liu et al., 2023, SAN, https://papers.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html

### 3. 손실함수 재설계

일반적인 MSE는 큰 가격 스케일과 평균 예측에 끌리기 쉽다. 금융 예측에서는 손실함수가 실제 의사결정과 맞지 않으면 loss는 줄어도 매매에 쓸 수 없는 예측이 나올 수 있다.

- 적용 후보: Huber + direction penalty, volatility-weighted loss, asymmetric loss, quantile loss, baseline-margin loss
- 확인 지표: sign agreement, downside error, persistence gap
- 참고: Elliott et al., 2005, Loss Functions in Time Series Forecasting, https://faculty.ucr.edu/~taelee/paper/lossfunctions.pdf
- 참고: Guo et al., 2025, A Novel Loss Function for Deep Learning Based Daily Stock Trading System, https://arxiv.org/abs/2502.17493

### 4. 예측 목표 재정의

다음 가격을 직접 맞히는 목표는 직전가 복사와 거의 같은 쉬운 문제로 바뀔 수 있다. 다음 단계에서는 target을 return, volatility-adjusted return, direction label, triple-barrier style label로 나눠 확인한다.

- 적용 후보: next log return, volatility-scaled return, sign target, neutral band classification, triple-barrier label
- 확인 지표: DA, MASE, sign agreement, neutral prediction share
- 참고: 금융 시계열 리뷰 문헌은 feature engineering, model choice, evaluation standard가 함께 설계되어야 함을 강조한다. TechScience CMES review, https://www.techscience.com/CMES/v139n1/55114/html

### 5. 기준선 대비 선택 규칙

이번 연구의 최저 방어선은 "딥러닝 모델이 단순 복사보다 나은가"이다. 다음 단계에서는 모델 선택 기준에 persistence baseline을 직접 넣는다.

- 적용 후보: persistence-margin early stopping, baseline-relative validation selection, collapse-aware ranking
- 확인 지표: persistence gap이 기준선 아래에 머무는 epoch 수, worst-regime persistence ratio
- 이유: loss가 예뻐도 baseline보다 못하면 연구적으로도 실전적으로도 채택하지 않는다.

## 6번 실험 단계

### Stage 0: 5번 결과 재현 확인

- 목적: 현재 로컬 노트북 출력과 동일한 15개 케이스 결과가 서버에서도 재현되는지 확인
- 수정 없음
- 실패 시: 데이터 split, seed, scaler, 출력 복원 경로부터 확인

### Stage 1: target 고정과 level target 제거

- 목적: `level_mse`를 기본 후보에서 제외하고 return target 중심으로 재정렬
- 수정: `return_huber`, `return_directional_hybrid`만 남겨 반복
- 이유: 5번에서 level target은 구조와 무관하게 실패했기 때문

### Stage 2: normalization ablation

- 목적: rolling scaling, robust scaling, RevIN-like scaling 중 무엇이 가장 안정적인지 확인
- 수정: 입력 normalization만 한 번에 하나씩 변경
- 이유: 비정상성을 한 번에 모델이 처리하게 두면 shortcut이 쉽게 생긴다.

### Stage 3: loss ablation

- 목적: Huber, directional hybrid, volatility-weighted directional loss, baseline-margin loss를 비교
- 수정: 손실함수만 한 번에 하나씩 변경
- 이유: 손실함수가 바뀌면 모델이 찾는 쉬운 답도 바뀐다.

### Stage 4: collapse-aware model selection

- 목적: validation loss 최저가 아니라 collapse와 persistence를 함께 통과한 epoch를 선택
- 수정: early stopping과 ranking 기준 변경
- 이유: loss 최저 epoch가 실제 예측 최선 epoch가 아닐 수 있다.

### Stage 5: 독립변수와 데이터마트 연결 전 최종 게이트

- 목적: 텍스트 독립변수, historical flow mart, 온체인/유동성 변수로 넘어가기 전 최소 안정성 기준 확정
- 통과 조건: persistence gap 기준선 하회, sign agreement 개선, collapse score 악화 없음
- 이유: 최적화 문제가 해결되지 않으면 독립변수를 붙여도 원인 해석이 흐려진다.

## 다음 코드 위치

- 실행 계획 코드: `test/models/6_optimization_stabilization_test.ipynb`
- diff 추적 미러: `test/models/6_optimization_stabilization_test.py`
- 이 코드는 기본적으로 실행 계획을 출력한다. 실제 학습 실행은 학교 서버나 승인된 원격 환경에서 `--run-stage`를 명시할 때만 수행한다.

## 보고서 작성 원칙

- 6번 보고서는 각 stage마다 `수정 전`, `수정 내용`, `좋은 그림 기준`, `실제 그림 해석`, `다음 수정 여부`를 기록한다.
- 같은 stage에서 여러 요소를 동시에 바꾸지 않는다.
- 결과 해석은 현재 로컬 또는 서버에서 실제 실행된 `.ipynb` 출력과 추출 이미지 기준으로만 작성한다.
