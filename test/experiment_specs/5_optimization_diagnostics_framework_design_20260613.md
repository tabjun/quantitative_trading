# 5. 최적화 학습 과정 진단 실험 설계서

작성일: 2026-06-13

## 초록

이 실험은 업비트 시계열에서 손실함수와 모델 구조가 실제 예측력으로 이어지는지, 아니면 직전가 복사나 0 수익률 예측 같은 쉬운 해로 붕괴하는지를 진단하기 위한 경량 실험이다.
목표는 성능 순위를 매기는 것이 아니라, 다음 단계의 독립변수 확장 전에 objective / target / head 설계가 안전한지 확인하는 것이다.

## 서론

비정상 금융 시계열은 train loss 하락만으로는 의미가 없다. validation이 같이 내려가는지, naive persistence를 이기는지, 방향성 우위가 있는지를 함께 봐야 한다.
따라서 이 실험은 예측 모델 평가라기보다 최적화 경로 진단이다.

## 방법론

### 데이터

- 데이터 원천: `data/upbit_data.db`
- 기본 대상: `btc_15m_advance`
- 입력: `log_return_1`, `return_4`, `realized_vol_16`, `hl_range_pct`, `volume_z_96`, `spread_proxy`
- 기본 길이: `seq_len=32`
- 기본 규모: `max_rows=35040`, `max_windows=1024`, `window_stride=4`

### 비교군

- `Linear`
- `LSTM`
- `GRU`

### 손실함수

- `MSE`
- `Huber`
- `directional hybrid`

### 진단 지표

- `train/validation loss index`
- `generalization gap`
- `gradient norm before clipping`
- `persistence gap`
- `collapse score`
- `near-zero return share`
- `sign agreement`

## 결과 읽는 법

- `train loss`와 `validation loss`가 같이 내려가야 한다.
- `validation - train gap`이 0보다 위로 올라가면 일반화 실패 가능성이 크다.
- `persistence gap`이 0보다 위면 naive persistence보다 못하다.
- `sign agreement`가 0.5 근처면 방향성 우위가 거의 없다.
- `collapse score`는 보조 지표이며 단독 결론으로 쓰지 않는다.

## 결론

이 실험의 합격 기준은 단순한 loss 하락이 아니다.
최소 기준은 validation 개선, naive persistence 돌파, 방향성 우위의 동시 확인이다.
이 기준을 못 넘으면 독립변수 추가보다 target, loss, head 재설계가 먼저다.

## 참고문헌 메모

- Hyndman & Koehler (2006): MASE 및 naive baseline 해석
- Geirhos et al. (2020): shortcut learning 개념
- Zeng et al. (2022): 단순 baseline의 강함
- Non-stationary Transformers / adaptive normalization / Koopman 기반 비정상 시계열 연구
- 금융/코인 텍스트 및 온체인 context 관련 최신 논문
