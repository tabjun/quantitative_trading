# 코인 맥락 변수와 비정상 시계열 최적화 경로 진단 브리프

작성일: 2026-06-11  
브랜치: `stock`

## 1. 문서 목적

이번 브리프는 현재 연구에서 별개로 보이지만 실제로는 하나의 설계 문제로 연결되는 두 가지 쟁점을 정리하기 위한 문서입니다.

1. 코인 데이터 분석에서 주식형 `증시 리포트 + 일반 뉴스` 중심 접근이 왜 불충분한가
2. 비정상 시계열 학습에서 관찰되는 실패가 왜 단순한 `기울기 소실(vanishing gradient)`보다는 `손실함수가 허용한 쉬운 해(shortcut / collapse)` 문제에 가까운가

핵심 결론은 다음과 같습니다.

- 코인 분석은 기업 펀더멘털 뉴스보다 거래소 미시구조, 온체인 흐름, 스테이블코인 공급, 레버리지/청산, 규제·보안 이벤트, 사회적 내러티브를 중심으로 독립변수를 설계해야 합니다.
- 최적화 진단은 단순 성능평가가 아니라, 학습 곡선을 통해 objective function이 `0 수익률 예측`, `lag-1 복사`, `평균 회귀형 flat output`을 얼마나 쉽게 허용하는지 확인하는 구조로 설계해야 합니다.

## 2. 문제 1: 코인 분석은 왜 주식형 텍스트 접근으로 부족한가

### 2.1 구조적 차이

주식은 기업 실적, 산업 리포트, 애널리스트 커버리지, 공시, 배당정책처럼 상대적으로 안정적인 가치 설명 변수가 존재합니다. 반면 코인은 다음과 같은 요인에 더 직접적으로 반응합니다.

- 거래소 유동성 및 미시구조
- 온체인 공급/이동/거래소 입출금
- 스테이블코인 발행과 환수
- 파생시장 레버리지, 펀딩비, 강제청산
- 상장/상폐/입출금 제한/해킹/브릿지 사고/규제 이벤트
- X, 텔레그램, 커뮤니티를 통한 내러티브 확산

즉, 코인에서 텍스트는 중요하지만 중심축이 `증시 리포트 감성`이 되어서는 안 되고, 사건의 원인을 해석하는 `context layer`로 배치되어야 합니다.

### 2.2 연구 설계에 대한 함의

과거 유사 구간 검색이나 독립변수 설계에서 다음 순서를 권장합니다.

1. 1차: 가격/거래량/변동성 형태로 shape retrieval
2. 2차: 온체인·유동성·파생·규제·보안·사회적 내러티브 context reranking
3. 3차: 현재 구간과 과거 구간의 원인 일치 여부를 설명

이 구조가 필요한 이유는, 비슷한 V자 반등이라도 원인이 다르면 동일 사건으로 취급하면 안 되기 때문입니다. 예를 들어 `청산 cascade 이후 반등`, `USDT mint 이후 유동성 반등`, `거래소 상장 효과`, `해킹 이후 기술적 반등`은 차트만 보면 비슷해도 맥락상 다른 사건입니다.

### 2.3 실무적으로 우선순위가 높은 코인 독립변수

- 거래소 레벨: 거래대금 급증, order-book imbalance, spread/depth proxy, 상장/상폐 공지
- 온체인 레벨: 거래소 순유입/순유출, active addresses, whale transfer, stablecoin flow
- 파생 레벨: open interest, funding rate, long/short skew, liquidation imbalance
- 거시/연계시장: BTC dominance, 달러, 금리, 나스닥, ETF flow
- 텍스트/내러티브: regulation, exploit, listing, partnership, macro shock, meme contagion

## 3. 문제 2: 이 현상은 왜 성능평가 이슈가 아니라 최적화 경로 진단 이슈인가

### 3.1 문제의 정확한 위치

이번 쟁점의 중심은 “어떤 모델이 더 잘 맞추는가”가 아닙니다. 더 중요한 질문은 다음과 같습니다.

> 비정상 금융 시계열에서 현재의 objective function이 모델을 어떤 해(solution)로 밀어 넣는가?

같은 데이터, 비슷한 아키텍처에서도 학습 곡선이 초기에 매우 빨리 안정화되는데 예측 결과는 의미 없는 경우가 있습니다. 이때 의심해야 할 것은 단순 gradient vanishing이 아니라, loss가 쉽게 보상하는 trivial solution입니다.

### 3.2 왜 기울기 소실만으로 설명하기 어려운가

기울기 소실이면 보통 다음 현상이 관찰됩니다.

- 학습이 거의 진행되지 않음
- 깊은 구조에서 gradient norm이 급격히 감소
- 초기 epoch부터 업데이트 자체가 약함

하지만 지금 진단하려는 현상은 종종 정반대입니다.

- loss는 초기에 잘 감소함
- 여러 아키텍처에서 비슷한 하강 곡선이 반복됨
- 그런데 예측은 `다음 수익률 ≈ 0`, `다음 가격 ≈ 현재 가격`처럼 수렴함

이는 objective mismatch 또는 shortcut learning의 성격이 더 강합니다. 즉, 모델이 “미래를 배웠다”기보다 “손실을 가장 쉽게 줄이는 방식”을 배운 것입니다.

### 3.3 통계적으로 보면 어떤 문제인가

이 현상은 다음 개념들과 연결됩니다.

- `non-stationarity`: 분포, 평균, 분산, 의존 구조가 시간에 따라 이동
- `conditional heteroskedasticity`: 변동성이 시간에 따라 군집적으로 변화
- `heavy tails`: 큰 변동이 드물지만 손실에 미치는 영향은 큼
- `objective mismatch`: downstream goal과 surrogate loss가 일치하지 않음
- `persistence dominance`: naive random-walk 또는 lag-1 복사가 surprisingly strong baseline이 됨
- `shortcut learning`: 구조를 학습하기보다 손쉬운 규칙으로 낮은 손실을 획득

비정상 자산 가격을 pointwise MSE/MAE로만 최적화하면, 조건부 평균에 가까운 해가 쉽게 선택됩니다. 특히 15분봉처럼 대부분의 구간에서 작은 변화가 반복되는 데이터는 `0 근처 수익률`이나 `직전 가격 복사`가 손실 측면에서 매우 매력적인 해가 됩니다.

### 3.4 따라서 봐야 하는 것은 “정확도”보다 “학습 곡선의 의미”

진단 실험에서 중요한 것은 단순 MAE/RMSE 순위가 아니라 다음 신호입니다.

- `loss`는 줄지만 `variance_ratio`가 0에 가까워지는가
- `zero_share`가 높아지며 거의 모든 예측이 0 근처로 수렴하는가
- `copy_alignment`가 과도하게 높아 현재 가격을 복사하는 경향이 강한가
- naive persistence와 비교해도 실질적 개선이 없는가

즉, 이번 실험은 evaluation leaderboard가 아니라 `optimization behavior analysis`입니다.

## 4. 이번에 추가한 실험 코드의 의도

이번에 추가한 코드는 다음 파일들입니다.

- `analysis/optimization_diagnostics.py`
- `test/models/5_optimization_diagnostics_test.ipynb`
- `test/models/5_optimization_diagnostics_test.py`

설계 의도는 전처리 변경이 아니라 `objective / head / architecture` 조합의 차이를 학습 곡선으로 비교하는 것입니다.

### 4.1 `objective_probe`

하나의 LSTM 계열 모델에 대해 아래 objective만 바꿉니다.

- `next_close_level + MSE`
- `next_log_return + MSE`
- `next_log_return + Huber`
- `next_log_return + volatility-weighted MSE`
- `next_log_return + directional hybrid loss`
- `next_log_return + tail-focus loss`

이 probe의 목적은 “어떤 target/loss 설계가 shortcut collapse를 가장 덜 허용하는가”를 보는 것입니다.

### 4.2 `architecture_probe`

같은 directional hybrid objective를 두고 아래 아키텍처를 비교합니다.

- Linear
- LSTM
- GRU
- TCN
- Transformer encoder

이 probe의 목적은 “objective를 고정해도 아키텍처에 따라 collapse 민감도가 달라지는가”를 확인하는 것입니다.

### 4.3 `full_matrix`

아키텍처와 objective 일부를 교차해, 보다 넓은 조합을 한 번에 볼 수 있게 했습니다.

## 5. 현재 연구에 대한 제안

### 5.1 코인 독립변수 설계

코인 맥락 변수는 아래처럼 계층화하는 것이 타당합니다.

- `market_microstructure_context`
- `onchain_flow_context`
- `stablecoin_liquidity_context`
- `derivatives_leverage_context`
- `regulation_security_context`
- `narrative_social_context`

텍스트는 이 계층 안에서 `event_type`, `source_type`, `coverage_count`, `shock_strength`, `lag`를 가진 구조화 변수로 쓰는 것이 바람직합니다.

### 5.2 최적화 실험 설계

학습 단계에서는 raw close level을 바로 맞추는 케이스를 “통제군”으로만 두고, 실제 연구 방향은 다음을 우선 검토하는 것이 좋습니다.

1. return target 기반 head
2. Huber 또는 weighted regression
3. directional hybrid penalty
4. tail / turning-point emphasis
5. 학습 곡선에서 collapse score, variance ratio, zero share 동시 모니터링

즉, 지금 단계에서 중요한 것은 “어떤 모델이 0.001 더 낮은 loss를 냈는가”보다 “어떤 objective가 잘못된 쉬운 해를 덜 허용하는가”입니다.

## 6. 참고문헌 및 활용 포인트

### A. 코인 독립변수 / 시장 맥락

1. Chi, Y., Chu, Q., & Hao, W. (2024). *Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets*. arXiv:2411.06327.  
   링크: https://arxiv.org/abs/2411.06327  
   활용 포인트: 거래소 순유입과 같은 on-chain flow가 BTC/ETH 수익률·변동성 예측력과 연결된다는 근거를 제공합니다.

2. Ante, L. (2025). *The Intraday Bitcoin Response to Tether Minting and Burning Events: Asymmetry, Investor Sentiment, And "Whale Alerts" On Twitter*. arXiv:2501.05232.  
   링크: https://arxiv.org/abs/2501.05232  
   활용 포인트: 스테이블코인 발행/소각과 같은 유동성 사건이 분 단위·시간 단위 수익률 반응을 유발할 수 있음을 보여줍니다.

3. *From On-chain to Macro: Assessing the Importance of Data Source Diversity in Cryptocurrency Market Forecasting* (2025).  
   링크: https://arxiv.org/html/2506.21246  
   활용 포인트: 기술지표, 온체인, 감성/관심도, 전통시장, 거시지표를 함께 쓰는 다원 데이터 소스 접근의 필요성을 보여줍니다.

4. Haritha, G. B., & Sahana, N. B. (2023). *Cryptocurrency Price Prediction using Twitter Sentiment Analysis*. arXiv:2303.09397.  
   링크: https://arxiv.org/abs/2303.09397  
   활용 포인트: 텍스트 감성은 단독 정답은 아니지만, 가격·트윗량·계정 속성과 결합될 때 설명력이 생길 수 있다는 예시입니다.

### B. 비정상 시계열 / 최적화 / 진단

5. Liu, Y., Wu, H., Wang, J., & Long, M. (2022). *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting*. arXiv:2205.14415.  
   링크: https://arxiv.org/abs/2205.14415  
   활용 포인트: 비정상성이 Transformer 계열의 성능을 크게 훼손할 수 있으며, 과도한 stationarization 또한 문제를 낳는다는 점을 설명합니다.

6. Liu, Z., Cheng, M., Li, Z., Huang, Z., Liu, Q., Xie, Y., & Chen, E. (2023). *Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective*. NeurIPS 2023.  
   간접 링크: https://arxiv.org/html/2409.20371v1  
   활용 포인트: 비정상 시계열에서는 단일 고정 정규화보다 시계열 slice별 적응형 정규화 관점이 중요하다는 흐름을 뒷받침합니다.

7. Liu, Y., Li, C., Wang, J., & Long, M. (2023). *Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors*. arXiv:2305.18803.  
   링크: https://arxiv.org/abs/2305.18803  
   활용 포인트: 비정상 시계열을 time-variant / time-invariant dynamics로 분리해 보는 동역학 관점을 제공합니다.

8. Yang, J., Hu, Y., Li, Y., Zhang, K., Ding, K., & Yu, P. S. (2026). *From Observations to States: Latent Time Series Forecasting*. arXiv:2602.00297.  
   링크: https://arxiv.org/abs/2602.00297  
   활용 포인트: 관측 공간의 pointwise error 최소화가 shortcut solution을 부를 수 있다는 최근 직접 근거입니다.

9. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). *Shortcut Learning in Deep Neural Networks*. arXiv:2004.07780.  
   링크: https://arxiv.org/abs/2004.07780  
   활용 포인트: 낮은 벤치마크 손실이 곧 올바른 구조 학습을 의미하지 않는다는 일반론적 배경을 제공합니다.

10. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022). *Are Transformers Effective for Time Series Forecasting?* arXiv:2205.13504.  
    링크: https://arxiv.org/abs/2205.13504  
    활용 포인트: 복잡한 시계열 아키텍처보다 단순 baseline이 더 강할 수 있다는 점을 보여주므로, 아키텍처 자체보다 objective와 진단 체계의 중요성을 상기시킵니다.

11. Hyndman, R. J., & Koehler, A. B. (2006). *Another Look at Measures of Forecast Accuracy*. International Journal of Forecasting, 22(4), 679-688.  
    링크: https://robjhyndman.com/papers/mase.pdf  
    활용 포인트: naive baseline 대비 scaled error인 `MASE` 해석의 표준 근거입니다. 이번 실험은 리더보드가 목적은 아니지만, naive persistence를 기준선으로 둬야 한다는 점을 설명하는 데 유용합니다.

## 7. 최종 정리

이번 연구에서 두 쟁점은 따로 존재하는 것이 아닙니다.

- 문제 1은 `무엇을 독립변수로 볼 것인가`의 문제이고,
- 문제 2는 `그 독립변수를 모델이 어떤 objective 아래에서 학습하게 할 것인가`의 문제입니다.

따라서 현재 단계에서 가장 중요한 실무적 태도는 다음과 같습니다.

1. 코인에서는 뉴스 일반론보다 사건 원인 중심의 context vector를 설계할 것
2. 성능 비교 전에 objective가 trivial solution을 허용하는지 학습 곡선으로 먼저 진단할 것
3. 모델 선택보다 `loss / head / diagnostic` 설계를 먼저 고정할 것

이 방향이 정리되면, 이후의 본격 실험은 단순 성능 비교가 아니라 “의미 있는 독립변수를, 잘못 붕괴하지 않는 objective 아래에서 학습시키는가”라는 더 단단한 연구 질문 위에서 진행될 수 있습니다.
