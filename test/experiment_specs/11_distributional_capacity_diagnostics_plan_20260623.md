# 11번 위험 이벤트·분포 예측 진단 계획

## 1. 결론부터 정리

사용자가 공유한 두 영상의 관점은 현재 연구에 반영할 가치가 있다. 다만 두 내용을 같은 종류의 전처리로 취급하면 안 된다.

- 불확실성 분포화는 **예측 출력과 손실함수의 확장**이다.
- Double Descent는 **모델 용량·표본 수·학습 시간과 일반화 오차의 관계를 확인하는 진단**이다.
- 둘 다 MinMax, Standard, Winsorize 같은 입력 전처리가 아니다.
- Black-Scholes 가격 공식을 15분 업비트 수익률 예측에 직접 적용하지 않는다.
- Black-Scholes에서 가져올 핵심은 “미래 가격은 하나의 확정값이 아니라 불확실한 분포”라는 관점이다.

10번은 objective와 validation-only ensemble로 다음 15분 수익률을 직접 예측하는 실험으로 그대로 보존한다. 11번은 10번을 덮어쓰지 않고, 향후 4시간 급변·하방 위험 확률과 직접 분포 예측을 경쟁 연구 질문으로 분리한다. Double Descent는 주 연구가 아니라 보조 진단으로 둔다.

## 2. 경쟁 가설과 병렬 실행

### 10번 점예측 가설

다음 15분 수익률의 방향과 크기를 직접 예측할 수 있다는 가설이다. persistence 대비 원화 MAE, 방향 정확도, 예측 분산을 평가한다.

### 11번 위험 이벤트 가설

정확한 다음 수익률은 어렵더라도 향후 16개 15분봉, 즉 약 4시간 안의 큰 움직임이나 하방 위험 발생 확률은 예측할 수 있다는 가설이다.

- `downside`: 미래 경로에서 가장 큰 누적 하락폭
- `absolute_move`: 미래 경로에서 가장 큰 절대 누적 움직임
- train 구간 위험 점수의 상위 10%를 event로 정의
- validation에서 분류 임계값을 고르고 test에서는 고정

두 코드는 서로의 결과를 읽지 않으므로 동시에 실행할 수 있다. 기본 자원 slot은 다음과 같다.

| 코드 | slot | GPU 상한 | batch | workers | torch threads |
|---|---|---:|---:|---:|---:|
| 10번 | `point_primary` | 52% | 32 | 2 | 8 |
| 11번 | `risk_secondary` | 36% | 24 | 2 | 6 |

두 프로세스 합계 GPU allocator 상한을 약 88%로 제한해 CUDA context와 일시적 메모리 공간을 남긴다. 한쪽에서 OOM이 발생하면 batch size를 절반으로 줄여 재시도한다.

## 3. 10번에 이미 들어간 것과 빠진 것

### 이미 들어간 것

10번에는 여러 seed와 모델의 예측을 평균·중앙값·validation 가중 평균으로 결합하는 ensemble이 있다. 또한 validation 오차로 conformal interval을 만들어 test 예측 주변의 범위를 표시한다.

이 방식은 “예측값 하나만 보여주지 말고 오차 범위도 함께 보자”는 불확실성 관점을 일부 반영한다. 특히 conformal interval은 특정 확률분포를 맞다고 가정하지 않고 validation 잔차를 이용해 경험적 범위를 만든다.

### 빠진 것

10번 모델은 여전히 다음 수익률 한 개를 출력한다. 구간은 학습 후에 붙인다. 다음 항목은 10번에 없다.

- 모델이 시점별 평균과 표준편차를 직접 출력하는 Gaussian likelihood
- 두꺼운 꼬리를 허용하는 Student-t likelihood
- 5%, 50%, 95% 분위수를 직접 출력하는 quantile regression
- likelihood 또는 quantile이 실제 빈도와 맞는지 보는 PIT·quantile calibration
- 모델 폭·표본 수·epoch를 체계적으로 바꾸는 Double Descent 곡선

따라서 10번은 “점예측을 안정화하고 여러 모델을 결합하는 실험”, 11번은 “예측분포와 모델 용량을 직접 진단하는 실험”으로 역할을 나눈다.

## 4. 위험 이벤트 예측이 무엇인가

점예측은 “다음 봉 수익률이 몇 퍼센트인가”를 묻는다. 위험 이벤트 예측은 “앞으로 4시간 안에 평소보다 큰 하락이나 큰 움직임이 발생할 확률이 얼마인가”를 묻는다.

예를 들어 실제 다음 수익률을 정확히 맞히지 못해도 하방 위험 확률을 0.80으로 예측하고 그 구간에서 실제 급락이 반복적으로 발생한다면, 신규 진입 회피·현금 비중 확대·포지션 축소에 활용할 수 있다.

평가 지표는 다음과 같다.

- `Average Precision`: 드문 위험 event를 높은 순위에 배치하는 정도다. 단순 기준은 event 발생률이다.
- `Brier score`: 예측 확률과 실제 0/1 결과의 제곱오차다. 작을수록 좋다. Train event 발생률을 모든 시점에 똑같이 내는 climatology 기준보다 좋아야 `Brier skill > 0`이 된다.
- `Reliability`: 0.7로 예측한 표본 중 실제 event가 약 70% 발생하는지 확인한다.
- `Risk-Coverage`: 확신이 높은 표본만 사용할 때 오류가 줄어드는지 확인한다.

## 5. 분포 예측이 무엇인가

### 쉬운 정의

점예측은 “다음 수익률은 0.2%다”처럼 숫자 하나를 답한다. 분포 예측은 “중앙 예상은 0.2%지만, 현재 변동성을 고려하면 대체로 -0.8%에서 1.1% 사이일 가능성이 높다”처럼 중심과 불확실성을 함께 답한다.

### 이번 연구에서 필요한 이유

업비트 15분 수익률은 극단값이 많고 시점에 따라 변동성이 달라진다. 9번 데이터의 수익률 첨도는 약 22.2로 정규분포보다 꼬리가 매우 두꺼웠다. 이런 데이터에 평균 오차만 줄이도록 요구하면 모델은 다음 두 쉬운 해로 갈 수 있다.

- 모든 예측을 0 근처로 줄여 평균 오차를 작게 만든다.
- 극단 구간을 따라가려다 평상시에도 지나치게 큰 진폭을 만든다.

분포 예측은 “중심을 어디에 둘지”와 “얼마나 불확실한지”를 함께 학습한다. 다만 구간을 출력한다고 중심 예측이 자동으로 좋아지는 것은 아니다. 점예측 MAE, 방향 정확도, 분산 비율과 구간 coverage·width를 함께 평가해야 한다.

## 6. 11번 분포 head

### Gaussian

모델은 위치 `mu`와 양수 척도 `sigma`를 출력하고 Gaussian negative log likelihood를 최소화한다.

```text
NLL = 0.5 * log(2*pi) + log(sigma) + 0.5 * ((y - mu) / sigma)^2
```

오차가 커질수록 더 큰 벌점을 받지만, 실제 업비트 수익률의 두꺼운 꼬리를 충분히 표현하지 못할 수 있다. Gaussian은 단순 기준선 역할을 한다.

### Student-t

모델은 위치 `mu`, 척도 `sigma`, 자유도 `nu`를 출력한다. 자유도가 작을수록 정규분포보다 꼬리가 두꺼워져 급등락에 더 높은 확률을 줄 수 있다.

예를 들어 Gaussian이 ±3 표준편차 밖의 움직임을 거의 불가능하다고 보는 반면, Student-t는 그런 사건을 드물지만 가능한 사건으로 남겨 둔다. 첨도가 높은 코인 수익률에서는 Gaussian보다 현실적인 후보지만, 척도를 무작정 크게 만들어 넓은 구간만 출력할 위험도 있다.

### Quantile

모델이 5%, 50%, 95% 분위수를 직접 출력하고 pinball loss를 최소화한다.

- 5% 분위수: 실제값이 이보다 작을 확률을 약 5%로 맞추려는 하한
- 50% 분위수: 중앙값
- 95% 분위수: 실제값이 이보다 작을 확률을 약 95%로 맞추려는 상한

특정 분포 모양을 정하지 않아도 되며 비대칭 구간을 만들 수 있다. 반면 세 분위수만으로 전체 확률밀도를 알 수는 없다.

### Conformal 보정

Gaussian, Student-t, quantile이 만든 원시 구간을 validation 자료의 실제 미포함 정도로 다시 넓힌다. 목표 coverage가 90%라면 test 실제값 약 90%가 구간 안에 들어오는지 확인한다.

coverage만 높이는 것은 충분하지 않다. 모든 경우를 포함하도록 구간을 아주 넓게 만들면 쉽게 100% coverage를 얻지만 의사결정에는 쓸 수 없다. 따라서 평균 interval width와 miss distance를 함께 본다.

## 7. 분포 평가 지표와 그래프

### Proper score

모델이 실제 관측값에 높은 확률을 주면서도 분포를 불필요하게 넓히지 않는지를 평가한다.

- Gaussian·Student-t: negative log likelihood
- Quantile: 5%, 50%, 95% pinball loss 평균

값이 작을수록 좋지만 서로 다른 score 종류의 절대값을 직접 순위 비교하기보다 같은 head 안에서 전처리·모델·seed를 비교한다.

### Coverage와 width

- coverage: 실제 수익률이 예측구간 안에 들어온 비율
- width: 상한과 하한의 평균 거리

90% 구간이라면 coverage가 0.90 부근이어야 한다. 같은 coverage라면 더 좁은 구간이 더 유용하다.

### PIT histogram

PIT는 관측된 실제값이 모델의 누적분포 안에서 어느 위치에 놓이는지 0과 1 사이 값으로 바꾼 것이다. 분포가 잘 맞으면 PIT 값이 0~1에 대체로 균일하게 퍼져야 한다.

- 양 끝에 몰리면 실제 극단값을 과소평가한 것이다.
- 중앙에 몰리면 구간이 필요 이상으로 넓을 가능성이 있다.
- 한쪽으로 치우치면 평균 또는 중앙값 편향이 남아 있다.

Quantile head는 전체 누적분포를 출력하지 않으므로 5%, 50%, 95%의 명목 분위수와 실제 누적 비율을 비교한다.

## 8. Double Descent가 무엇인가

### 쉬운 정의

전통적인 설명에서는 모델이 복잡해질수록 처음에는 성능이 좋아지다가 과적합 구간에서 test 오차가 다시 커지는 U자형 곡선을 예상한다.

Double Descent에서는 모델 용량이 더 커져 학습자료를 거의 완전히 맞추는 interpolation threshold 근처에서 test 오차가 크게 나빠졌다가, 그보다 훨씬 큰 과매개변수 모델에서 오히려 다시 낮아질 수 있다.

즉 “큰 모델은 항상 나쁘다”도 아니고 “큰 모델은 항상 좋다”도 아니다. 데이터량과 파라미터 수의 상대적 위치에 따라 위험 구간이 생길 수 있다는 뜻이다.

### 왜 현재 연구에 관련 있는가

현재 연구는 width 64~128의 shallow-but-wide 구조를 주로 사용한다. 9번과 10번에서 모델별 평탄화와 분산 폭주가 크게 달랐기 때문에, hidden width를 임의로 96에 고정하면 다음 가능성을 놓칠 수 있다.

- width 96이 특정 모델에서 interpolation 위험 구간에 가깝다.
- 더 작은 모델은 underfit이지만 더 큰 모델은 다시 안정화된다.
- 데이터 window 수를 늘리면 위험 구간의 위치가 이동한다.
- epoch가 길어질수록 같은 모델이 interpolation 구간을 통과한다.

### 전처리인가

아니다. 전처리를 고정하고 다음 축을 바꾸어야 Double Descent를 해석할 수 있다.

- model-wise: hidden width와 parameter count
- sample-wise: 학습 window 수
- epoch-wise: 학습 epoch

전처리까지 동시에 바꾸면 어느 변화가 곡선을 만든 것인지 알 수 없으므로 11번 기본 capacity suite는 9번 선별 후보인 `seasonal_diff16`을 고정한다.

## 9. 11번 suite

### risk_event_probe

- horizon: 16개 15분봉, 약 4시간
- event: downside, absolute move
- 모델: Linear, PatchTSTLike
- 전처리: seasonal_diff16, winsor_025
- seed: 42, 137, 2026
- 기본 24개 case

각 case마다 학습 곡선, 시점별 위험 확률, reliability diagram, precision-recall curve, selective risk-coverage, 실현 위험 점수와 예측 확률을 표시한다.

### distribution_probe

- 모델: Linear, PatchTSTLike
- 전처리: seasonal_diff16, winsor_025
- head: Gaussian, Student-t, Quantile
- seed: 42, 137, 2026
- 기본 36개 case

각 case마다 학습 곡선, 실제·중앙 예측·원시/보정 구간, interval width와 실제 오차, calibration scatter, PIT 또는 quantile calibration, 불확실성 크기와 표준화 잔차를 출력한다.

### model_capacity_probe

- 모델: Linear, PatchTSTLike
- width: 16, 24, 32, 48, 64, 96, 128, 192, 256
- seed: 42, 137, 2026
- 표본 수: 전체 train window

파라미터 수에 따른 train loss, validation loss, persistence 대비 test MAE를 그린다.

### sample_size_probe

- 기본 모델: 첫 번째 지정 모델
- width 전체
- train fraction: 25%, 50%, 75%, 100%
- validation과 test 구간은 고정

같은 모델 용량이라도 학습 표본 수가 달라질 때 interpolation 위험 구간이 이동하는지 본다.

### epoch_probe

- width: 32, 96, 256
- epoch별 train·validation·test diagnostic loss 표시

test 곡선은 현상 진단용으로만 사용하고 checkpoint 선택에는 사용하지 않는다. 최종 모델 상태 선택은 validation loss 기준이다.

### full

네 suite를 모두 구성한다. 시간이 매우 오래 걸리므로 한 번에 시작하지 않고 각 suite 결과를 확인한 뒤 수행한다.

## 10. 서버 병렬 실행 순서

### 커널 A: 10번

첫 번째 venv 커널에서 10번 노트북을 실행한다. 기본 slot은 `point_primary`다.

### 커널 B: 11번

두 번째 venv 커널에서 11번 노트북을 실행한다. 기본 suite와 slot은 각각 `risk_event_probe`, `risk_secondary`다.

### case 구성 확인

```bash
python test/models/11_distributional_capacity_diagnostics_test.py \
  --suite risk_event_probe \
  --parallel-slot risk_secondary \
  --max-cases 6 \
  --dry-run
```

### 위험 이벤트 병렬 본실험

```bash
python test/models/11_distributional_capacity_diagnostics_test.py \
  --suite risk_event_probe \
  --parallel-slot risk_secondary \
  --models Linear,PatchTSTLike \
  --event-kinds downside,absolute_move \
  --preprocessings seasonal_diff16,winsor_025 \
  --seeds 42,137,2026 \
  --horizon 16 \
  --event-quantile 0.90 \
  --epochs 20 \
  --max-windows 4096 \
  --max-cases 24 \
  --continue-on-failure
```

### 위험 이벤트 통과 후 분포 본실험

```bash
python test/models/11_distributional_capacity_diagnostics_test.py \
  --suite distribution_probe \
  --parallel-slot risk_secondary \
  --epochs 20 \
  --max-windows 4096 \
  --continue-on-failure
```

### 모델 용량 확인

```bash
python test/models/11_distributional_capacity_diagnostics_test.py \
  --suite model_capacity_probe \
  --preprocessings seasonal_diff16 \
  --capacity-epochs 30 \
  --max-windows 4096 \
  --continue-on-failure
```

### 표본 수 확인

```bash
python test/models/11_distributional_capacity_diagnostics_test.py \
  --suite sample_size_probe \
  --models PatchTSTLike \
  --preprocessings seasonal_diff16 \
  --capacity-epochs 30 \
  --max-windows 4096 \
  --continue-on-failure
```

### Epoch-wise 확인

```bash
python test/models/11_distributional_capacity_diagnostics_test.py \
  --suite epoch_probe \
  --models Linear,PatchTSTLike \
  --epoch-widths 32,96,256 \
  --capacity-epochs 60 \
  --max-windows 4096 \
  --continue-on-failure
```

## 11. 실행 판단 기준

위험 이벤트 가설은 다음 조건을 함께 본다.

1. Average Precision이 event 발생률보다 반복적으로 높다.
2. Brier score와 calibration error가 단순 event-rate 확률보다 낮다.
3. 여러 seed에서 같은 전처리·모델 경향이 반복된다.
4. coverage를 50%로 줄여 확신이 높은 경우만 사용할 때 classification error가 감소한다.
5. 향후 서비스의 진입 회피 또는 포지션 축소 규칙으로 연결할 수 있다.

분포 모델은 다음 조건을 함께 만족해야 후속 후보로 남긴다.

1. copy-risk ratio가 1에 가까워지거나 1 미만이다.
2. direction accuracy와 variance ratio가 평탄화 또는 폭주가 아닌 범위에 있다.
3. 90% interval coverage가 0.90 부근이다.
4. 같은 coverage에서 interval width가 과도하게 넓지 않다.
5. 고변동 구간 coverage가 저변동 구간보다 심하게 무너지지 않는다.
6. 여러 seed에서 같은 경향이 반복된다.

Double Descent는 다음 조건이 있어야 관찰되었다고 판단한다.

1. parameter count 증가에 따라 test error가 단순 잡음이 아니라 상승 후 재하락한다.
2. train error 또는 interpolation 지표와 굴곡 위치가 연결된다.
3. 여러 seed에서 비슷한 위치 또는 방향이 반복된다.
4. sample fraction을 바꾸면 위험 구간 위치가 논리적으로 이동한다.
5. 한 번의 우연한 width 우승을 Double Descent라고 부르지 않는다.

## 12. 10번과 11번 방향 선택

10번과 11번은 target이 다르므로 MAE 하나로 직접 비교하지 않는다.

- 10번 통과: 여러 seed에서 copy-risk ratio 1 미만, 방향 정확도와 분산이 안정적
- 11번 통과: Average Precision·Brier·calibration·selective risk가 기준선보다 안정적

둘 다 통과하면 점예측은 예상수익, 위험 이벤트는 포지션 크기와 거래 거부에 사용한다. 10번만 실패하면 직접 점예측 연구선을 종료하고 11번 위험 관리 방향을 주 연구로 전환한다. 11번도 실패하면 OHLCV만으로 부족하다는 결론을 내리고 오더북·텍스트·다종목 요인을 추가한다.

## 13. 추가로 반영할 논문과 우선순위

### 지금 11번에 직접 반영

- Belkin et al., *Reconciling modern machine-learning practice and the classical bias-variance trade-off*  
  https://arxiv.org/abs/1812.11118
- Nakkiran et al., *Deep Double Descent: Where Bigger Models and More Data Hurt*  
  https://arxiv.org/abs/1912.02292
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*  
  https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- Romano et al., *Conformalized Quantile Regression*  
  https://arxiv.org/abs/1905.03222
- Jensen et al., *Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting*  
  https://arxiv.org/abs/2202.08756
- Salinas et al., *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*  
  https://arxiv.org/abs/1704.04110

### 11번 결과가 유효할 때 다음 번호에서 검토

- TimeGrad: 분포를 간단한 Gaussian·Student-t로 제한하지 않고 diffusion sampling으로 생성  
  https://arxiv.org/abs/2101.12072
- TimeDiff: 비자동회귀 diffusion 기반 시계열 예측  
  https://arxiv.org/abs/2306.05043
- TFT: 외생변수와 multi-horizon quantile 예측을 함께 다루는 구조  
  https://arxiv.org/abs/1912.09363

Diffusion 계열은 계산량과 구현 복잡도가 커서 현재 11번의 첫 단계에 넣지 않는다. 먼저 Linear와 PatchTSTLike에서 Gaussian·Student-t·quantile이 실제로 coverage와 sharpness를 개선하는지 확인한 뒤 확장한다.

## 14. 영상의 프로젝트 적용 범위

- 불확실성 영상: https://www.youtube.com/watch?v=99BHnu64pu8
- Double Descent 영상: https://www.youtube.com/watch?v=5ruiphjlOwo

영상은 연구 질문을 이해하는 출발점으로 사용하고, 구현과 보고서의 방법론 근거는 위 원 논문을 기준으로 한다. 영상에서 설명한 수식을 그대로 복제하기보다 현재 업비트 수익률의 heavy-tail, 비정상성, persistence 실패 문제에 맞게 분포 head와 용량 진단으로 번역한다.

## 15. 산출물 원칙

- 결과 원본은 `test/models/11_distributional_capacity_diagnostics_test.ipynb`의 inline 출력이다.
- 모든 그림은 `plt.show()`로만 표시한다.
- 서버에 PNG, CSV, Markdown 결과 파일을 자동 저장하지 않는다.
- 보고서 이미지는 실행 완료 후 로컬 `test/scripts/extract_notebook_images.py`로 노트북 출력에서 추출한다.
