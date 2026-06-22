# 9번 전처리·불확실성·용량 진단 계획

## 1. 왜 9번을 별도로 만드는가

8번 breadth probe에서는 14개 모델 모두 persistence baseline을 이기지 못했다. 다만 실패 형태는 같지 않았다.

- TimesNetLike, NLinearLike, DLinearLike, TCN은 예측 분산이 실제의 1% 미만으로 줄어 0수익률 근처에 머물렀다.
- LSTM, GRU, AutoformerLike는 실제보다 큰 예측 분산을 만들어 폭주했다.
- Linear와 PatchTSTLike는 MAE는 더 컸지만 방향과 변동 정보를 상대적으로 더 많이 보존했다.

따라서 다음 단계는 모델 이름을 더 추가하는 것이 아니라, 전처리와 분포 처리 방식이 두 가지 collapse를 어떻게 바꾸는지 촘촘하게 확인해야 한다. 기존 8번의 의미를 바꾸지 않기 위해 새 번호 9번으로 분리한다.

## 2. 두 영상의 연구적 적용

### 2.1 Black–Scholes 영상

- 영상: [수학자들이 얼마나 돈을 벌고 싶은지 감도 안옴](https://www.youtube.com/watch?v=99BHnu64pu8)
- 적용할 핵심: 불확실한 미래를 단일 가격이 아니라 확률분포와 변동성으로 표현한다.
- 적용하지 않을 부분: 코인 수익률에 Black–Scholes의 로그정규 가정을 그대로 강제하지 않는다.

9번 반영 항목:

- seed ensemble
- conformal prediction interval
- interval coverage
- interval width
- interval miss distance
- seed disagreement
- 변동 구간별 interval width

### 2.2 Double Descent 영상

- 영상: ["교수님, 전공책이 틀렸습니다" | 더블디센트](https://www.youtube.com/watch?v=5ruiphjlOwo)
- 적용할 핵심: 모델 복잡도와 일반화 오차의 관계를 단순 U자로 가정하지 않는다.

9번 반영 항목:

- hidden width: 24, 48, 64, 96, 128, 192, 256
- parameter count 기록
- seed: 42, 137, 2026
- epoch별 train/validation loss
- 모델 크기 대비 test MAE
- 모델 크기 대비 collapse 지표
- seed별 예측 분산

공유 대화의 “불확실성을 모델링해 편향을 줄인다”는 표현은 조금 더 정확히 구분해야 한다. 불확실성 구간을 출력한다고 모델의 구조적 편향이 자동으로 줄어드는 것은 아니다. 다만 점예측 하나만 보고 잘못된 확신을 갖는 의사결정 편향을 줄이고, seed·표본·분포 이동에 대한 예측 변동성을 관찰할 수 있다.

## 3. 전처리 실험 축

### 3.1 극단값 처리

- `winsor_005`: 양 끝 0.5% 절단
- `winsor_01`: 양 끝 1% 절단
- `winsor_025`: 양 끝 2.5% 절단
- `hampel_3`: rolling median과 MAD를 이용한 국소 이상치 치환

비교 목적:

- 과도한 극단값이 gradient와 출력 분산을 폭주시켰는지 확인
- 절단이 너무 강해 방향 정보까지 제거하는지 확인

### 3.2 Heavy-tail과 비선형 스케일

- `asinh_robust`
- `signed_log1p`
- `winsor_01+asinh_robust`
- `hampel_3+asinh_robust`

`asinh`는 0 근처에서는 거의 선형이고 큰 절댓값에서는 로그처럼 압축한다. 수익률 부호를 보존하면서 두꺼운 꼬리를 완화할 수 있다. 최신 NoRIN 역시 affine normalization만으로는 heavy-tail과 skewness를 바꾸기 어렵다는 문제를 제기하고 비선형 가역 변환을 사용한다.

### 3.3 추세와 저주파 제거

- `first_diff`
- `seasonal_diff4`
- `seasonal_diff16`
- `ema_residual_5`
- `ema_residual_9`
- `ema_residual_17`
- `linear_detrend`
- `median_residual_5`

비교 목적:

- 이동 추세가 모델을 직전가 복사로 유도했는지 확인
- 차분이 신호까지 제거해 0수익률 collapse를 강화하는지 확인
- EMA와 median residual의 강건성 차이를 확인

### 3.4 주파수 분해

- `frequency_highpass_1`
- `frequency_highpass_3`
- `frequency_highpass_5`
- `frequency_bandpass`
- `winsor_01+frequency_highpass_3`

FAN과 FredNormer가 제안한 문제의식처럼, 평균·분산 기반 정규화만으로는 동적 주기와 저주파 이동을 구분하기 어렵다. 9번은 완전한 FAN 재현이 아니라, 어떤 주파수 제거 강도가 현재 코인 데이터에서 정보를 보존하는지 보는 대표 실험이다.

### 3.5 변동성 조정

- `volatility_scale`
- `volatility_scale+asinh_robust`

변동성 군집이 강한 구간에서 동일한 크기의 수익률 오차를 동일하게 취급하지 않도록 국소 변동성으로 입력 스케일을 조정한다. 결과는 반드시 원래 KRW 가격으로 복원해 평가한다.

### 3.6 조합 전처리

- `winsor_01+ema_residual_9`
- `asinh_robust+ema_residual_9`
- `linear_detrend+asinh_robust`
- `seasonal_diff4+asinh_robust`

조합은 “많이 적용할수록 좋다”는 가정이 아니라, 극단값·추세·heavy-tail 문제를 분리했을 때 collapse 지표가 실제로 개선되는지 확인하기 위한 것이다.

## 4. Suite 구성

### `preprocessing_matrix`

- 28개 전처리 pipeline
- Linear, PatchTSTLike, TimesNetLike, AutoformerLike
- 기본 112 cases
- 모델별로 평탄화·정보 보존·폭주 대조군을 함께 둔다.

### `uncertainty_probe`

- 대표 전처리 4개
- 최대 3개 모델
- seed 3개
- seed ensemble과 conformal interval을 결합

### `capacity_probe`

- 최대 3개 모델
- width 7단계
- seed 3개
- parameter count와 test error 곡선을 출력

### `full`

- 전처리 × 모델 × seed 전체 조합
- OOM과 장시간 실행을 피하기 위해 `--max-cases`로 stage를 나눈다.

## 5. 중간 시각화

실행 과정에서 다음 그림을 모두 notebook inline output으로 출력한다.

1. 원시 BTC 가격
2. 15분 로그수익률
3. 수익률 histogram
4. 정규분포 QQ 진단
5. rolling realized volatility
6. rolling mean/std drift
7. train/validation/test 시간 분할
8. 전처리 전후 시계열 gallery
9. 전처리 후 분포 gallery
10. 전처리 후 power spectrum gallery
11. epoch별 train/validation loss
12. gradient mean/max norm
13. learning rate
14. 실제/예측 로그수익률
15. 실제/예측 KRW 가격과 persistence
16. 실제 대 예측 scatter
17. conformal prediction interval
18. seed disagreement 대 실제 오차
19. copy-risk leaderboard
20. variance ratio 대 direction accuracy
21. parameter count 대 test MAE
22. preprocessing × model heatmap

`savefig()`는 사용하지 않는다. 노트북 출력이 분석 원본이며 보고서 이미지는 로컬 후처리 도구로 추출한다.

## 6. 판단 기준

좋은 전처리는 단순히 MAE를 낮추는 것이 아니다.

- copy-risk ratio < 1
- variance ratio가 0 또는 과도한 값에서 벗어남
- direction accuracy가 우연 수준보다 안정적으로 높음
- near-zero share 감소
- seed 간 예측 차이 감소
- 명목 90% interval의 실제 coverage가 90%에 가까움
- 같은 coverage에서 interval width가 더 좁음

## 7. 실행 예시

작은 전처리 점검:

```bash
python test/models/9_preprocessing_uncertainty_diagnostics_test.py \
  --suite preprocessing_matrix \
  --models Linear,PatchTSTLike \
  --preprocessings none,winsor_01,asinh_robust,winsor_01+asinh_robust,ema_residual_9 \
  --epochs 3 \
  --max-windows 1024 \
  --max-cases 10 \
  --continue-on-failure
```

전체 전처리 matrix:

```bash
python test/models/9_preprocessing_uncertainty_diagnostics_test.py \
  --suite preprocessing_matrix \
  --epochs 12 \
  --max-windows 4096 \
  --max-cases 112 \
  --continue-on-failure
```

불확실성 점검:

```bash
python test/models/9_preprocessing_uncertainty_diagnostics_test.py \
  --suite uncertainty_probe \
  --models Linear,PatchTSTLike,TimesNetLike \
  --seeds 42,137,2026 \
  --epochs 12 \
  --max-cases 36 \
  --continue-on-failure
```

Double Descent 용량 점검:

```bash
python test/models/9_preprocessing_uncertainty_diagnostics_test.py \
  --suite capacity_probe \
  --models Linear,PatchTSTLike,ModernTCNLike \
  --widths 24,48,64,96,128,192,256 \
  --seeds 42,137,2026 \
  --max-cases 63 \
  --continue-on-failure
```

## 8. 참고 논문

- [Belkin et al., Reconciling modern machine-learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)
- [Nakkiran et al., Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [Kim et al., Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p)
- [Fan et al., Dish-TS](https://arxiv.org/abs/2302.14829)
- [Ye et al., Frequency Adaptive Normalization](https://arxiv.org/abs/2409.20371)
- [Piao et al., FredNormer](https://arxiv.org/abs/2410.01860)
- [Zhang and Xiao, NoRIN](https://arxiv.org/abs/2605.10823)
- [Jensen et al., Ensemble Conformalized Quantile Regression](https://arxiv.org/abs/2202.08756)
- [Prinzhorn et al., Conformal time series decomposition](https://arxiv.org/abs/2406.16766)

NoRIN은 2026년 5월 공개된 매우 최근 preprint이므로 확정된 표준 방법처럼 취급하지 않는다. 다만 현재 데이터의 높은 첨도와 affine normalization의 한계를 연결하는 실험 가설로는 유용하다.
