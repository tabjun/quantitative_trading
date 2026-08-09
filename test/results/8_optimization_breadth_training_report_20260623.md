# 8번 최적화 breadth training 결과 보고서

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
