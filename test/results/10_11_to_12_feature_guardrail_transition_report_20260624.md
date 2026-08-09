# 10·11번 결과 기반 12번 feature guardrail 전환 보고서

## 1. 결론부터

지금은 모델 구조를 더 늘릴 단계가 아니다. 10번과 11번 결과를 함께 보면, 최적화가 쉬운 해로 붕괴하는 문제는 완전히 해결되지는 않았지만 **다음 최적화 단계로 넘어갈 만큼 완화되었다**고 판단할 수 있다.

다음 단계는 다음처럼 잡는 것이 맞다.

```text
10번 balanced_composite 점예측
  + 11번 absolute_move 위험확률 guardrail
  + 코인 독립변수 조합별 ablation
  -> 어떤 feature group을 데이터마트로 승격할지 결정
```

즉 12번은 “새 알고리즘을 최고 수준으로 다시 설계하는 실험”이 아니다. 10번과 11번에서 살아남은 구조를 유지한 채, **코인이라는 종목 특성에 맞는 독립변수 조합이 점예측 붕괴와 MDD를 줄이는지** 보는 실험이다.

## 2. 10번에서 무엇을 가져갈 것인가

10번의 핵심은 `huber`와 `balanced_composite`의 차이다.

`huber`는 큰 오차의 영향을 줄이는 안정적인 손실함수다. 금융 시계열처럼 급등락이 많은 데이터에서는 큰 오차 하나가 학습을 망가뜨리는 것을 막아 준다. 하지만 이번 데이터에서는 이 안정성이 오히려 “작게만 예측하면 손실이 크게 튀지 않는다”는 쉬운 해로 이어질 수 있었다.

반면 `balanced_composite`는 단순 Huber 오차에 더해 방향성, 분산, 상관, tail, regime, anti-collapse 항을 함께 본다. 쉽게 말하면 “오차만 작게 만들지 말고, 예측이 실제 움직임처럼 어느 정도 흔들리고 방향도 맞추도록” 압력을 주는 objective다.

10번의 상위 결과는 다음과 같이 읽어야 한다.

| 관찰 | 의미 |
|---|---|
| best single model MAE가 persistence보다 큼 | 아직 점예측 단독 성공은 아님 |
| `balanced_composite`가 `huber`보다 방향과 분산을 더 남김 | 쉬운 해 collapse가 줄어듦 |
| validation-only ensemble이 copy-risk를 1.02 수준까지 낮춤 | 기준선에 근접했지만 아직 통과는 아님 |
| 그래프에서 예측값이 여전히 평평한 구간이 있음 | 독립변수/전처리/horizon 개선 여지가 있음 |

따라서 10번을 버리는 것이 아니라, 다음 단계의 **점예측 기본 토대**로 쓰는 것이 맞다.

## 3. 11번에서 무엇을 가져갈 것인가

11번은 다음 가격을 직접 맞히는 실험이 아니라, 향후 4시간 안에 큰 움직임이 발생할 확률을 보는 실험이다. 여기서 가장 중요한 target은 `absolute_move`다.

`absolute_move`는 상승이든 하락이든 큰 움직임이 있었는지를 본다. 예를 들어 앞으로 4시간 동안 BTC가 위아래 어느 쪽으로든 크게 움직이면 1, 조용하면 0으로 보는 방식이다. 방향은 알려 주지 않지만, 위험한 구간을 피하는 데 도움이 된다.

11번에서 `PatchTSTLike + seasonal_diff16 + absolute_move`는 다음 결과를 보였다.

| 지표 | 값 | 해석 |
|---|---:|---|
| Average Precision | 약 0.608 | 위험 이벤트 순서를 기준선보다 잘 나눔 |
| AP lift | 약 1.648 | 단순 발생률 대비 약 1.65배 나은 정렬 |
| ROC AUC | 약 0.716 | 위험/비위험 구분 능력이 있음 |
| Brier skill | 약 0.321 | 평균 발생률만 예측하는 것보다 확률 예측이 나음 |
| ECE | 약 0.094 | 확률 보정은 추가 개선 필요 |

따라서 11번은 점예측을 대체하는 모델이 아니라, 점예측을 **언제 믿지 말아야 하는지** 알려주는 guardrail로 쓰는 것이 적절하다.

## 4. 왜 10번만 보면 부족한가

10번 그래프를 보면 total objective는 안정적으로 내려간다. 이것은 좋은 신호다. 하지만 금융 시계열에서는 loss가 내려간다는 사실만으로 충분하지 않다. 모델이 실제 변동을 배운 것이 아니라, 다음 수익률을 거의 0으로 말하는 방식으로도 loss를 줄일 수 있기 때문이다.

따라서 10번만 보면 다음 문제가 남는다.

- 예측이 평평해지는 구간이 여전히 있다.
- persistence를 아직 넘지 못한다.
- 방향성이 52~53% 수준이라 단독 신호로는 약하다.
- 독립변수가 부족해서 모델이 쓸 정보가 제한되어 있을 가능성이 크다.

이 문제는 모델 구조를 더 늘린다고 바로 풀릴 가능성이 낮다. 오히려 코인에 맞는 독립변수를 넣고, 그 정보가 collapse를 줄이는지 보는 편이 더 직접적이다.

## 5. 왜 11번을 guardrail로 붙이는가

점예측은 “얼마나 오를지/내릴지”를 말한다. 위험확률은 “앞으로 크게 흔들릴 가능성이 높은지”를 말한다. 둘은 서로 다른 정보다.

예를 들어 10번 점예측이 다음 15분 수익률을 양수로 봤다고 하자. 이때 11번 위험확률이 낮으면, 해당 신호는 비교적 조용한 구간의 상승 후보로 남길 수 있다. 반대로 11번 위험확률이 높으면, 점예측이 양수여도 급변 위험이 높으므로 진입을 막거나 포지션을 줄일 수 있다.

이 결합은 다음 구조다.

| 10번 점예측 | 11번 위험확률 | 행동 후보 |
|---|---|---|
| 양수, 비용 초과 | 낮음 | 진입 후보 |
| 양수, 비용 초과 | 높음 | 관망 또는 축소 |
| 약한 양수 | 낮음 | 소량 또는 관망 |
| 음수 | 높음 | 진입 금지 |

이 방식은 점예측 모델이 아직 완벽하지 않아도 활용할 여지를 만든다. 특히 이 프로젝트의 핵심 목적이 “최고 수익률”보다 “최하방 방어, MDD 최소화”라면 11번의 위험확률은 꽤 중요한 출력이다.

## 6. 12번에서 테스트할 코인 독립변수

12번은 다음 feature group을 비교한다.

| feature group | 포함 변수 예 | 목적 |
|---|---|---|
| `ohlcv_core` | log return, realized volatility, volume/value z-score | 기준선 |
| `coin_liquidity_micro` | turnover, range, Amihud-style illiquidity, volume acceleration | 코인 유동성 변화가 예측 붕괴를 줄이는지 |
| `coin_volatility_regime` | vol ratio, downside/upside vol, drawdown, tail return | 위험 gate와 맞는 regime 정보가 있는지 |
| `coin_momentum_reversal` | RSI, MACD, EMA gap, trend strength, reversal | 점예측 방향성이 개선되는지 |
| `coin_orderflow_proxy` | candle body, upper/lower wick, close location, signed volume/value | 실제 order book 없이도 단기 매수·매도 압력 proxy가 도움이 되는지 |
| `coin_multitimeframe_structure` | 4/16/64/192 window return, volatility, z-score, trend | 단일 15분봉보다 여러 시간축 구조가 더 안정적인지 |
| `coin_shock_event` | return shock, volume shock, value shock, range shock, jump-reversal | 이벤트성 급변과 되돌림이 위험 gate와 맞물리는지 |
| `coin_attention_proxy` | volume/value/range shock 기반 관심도 proxy | 검색·소셜 데이터 전 단계에서 attention 신호가 유효한지 |
| `coin_calendar_cycle` | hour/day sin/cos, 한국/미국 시간대 proxy | 24시간 거래 시장의 시간대 효과 |
| `coin_text_context` | sentiment, shock, topic/event count | 텍스트 mart가 있으면 위험확률 개선에 쓰이는지 |
| `coin_cross_market` | ETH/XRP/SOL 등 cross return | 시장 전체 흐름이 BTC 예측에 도움 되는지 |
| `coin_macro_proxy` | DXY/VIX/금리/주가지수/환율 등 컬럼이 있을 때 | BTC가 위험자산처럼 움직이는 레짐을 설명하는지 |
| `coin_onchain_proxy` | active address, exchange flow, whale, SOPR, MVRV 등 컬럼이 있을 때 | 코인 고유 수급·네트워크 활동이 도움이 되는지 |
| `coin_derivatives_proxy` | funding, open interest, basis, liquidation 등 컬럼이 있을 때 | 레버리지·청산 압력이 급변 예측에 도움이 되는지 |
| `coin_search_social_dev` | Google Trends, YouTube, Twitter/X, Reddit, GitHub 등 컬럼이 있을 때 | 관심도·커뮤니티·개발자 활동이 선행 정보인지 |
| `coin_full_available` | 현재 사용 가능한 모든 feature | feature를 많이 넣는 것이 오히려 노이즈인지 확인 |

이 선택은 기존 문헌 흐름과도 맞다. 코인 예측 연구들은 보통 OHLCV/기술지표만이 아니라 유동성, sentiment/search, macro, cross-market, on-chain/orderflow를 함께 검토한다. 여기에 실무형 자료에서 자주 강조되는 multi-timeframe, order flow, liquidity sweep, volume shock 관점도 넣었다. 다만 현재 저장소에 아직 on-chain, funding/open interest, orderbook, Google Trends, YouTube/SNS, GitHub activity mart가 없을 수 있으므로, 12번에서는 OHLCV 기반 proxy를 먼저 계산하고 외부 컬럼이 있으면 자동으로 추가 group을 등록한다.

## 7. 12번 그래프는 어떻게 읽을 것인가

12번 노트북은 case별로 다음 그래프를 출력한다.

### 7.1 Point forecast graph

- 데이터: BTC 15분봉에서 만든 feature group별 입력
- 모델: 10번의 `Linear` 또는 `PatchTSTLike` point branch
- x축: test time index
- y축: 다음 15분 로그수익률
- 목적: 예측값이 실제 수익률을 어느 정도 따라가는지, 또는 다시 0 근처로 평평해지는지 확인
- 좋은 그림: 예측선이 실제보다 작더라도 방향 변화와 일부 진폭을 따라간다.
- 나쁜 그림: 예측선이 거의 0에 붙어 있거나 실제와 무관하게 흔들린다.

### 7.2 Risk guardrail graph

- 데이터: 같은 feature group에서 만든 향후 4시간 risk window
- 모델: 11번의 `PatchTSTLike` risk branch
- x축: test time index
- y축: `absolute_move` 위험확률
- 목적: 위험한 구간에서 확률이 올라가는지 확인
- 좋은 그림: 실제 큰 움직임 주변에서 확률이 올라가고, 낮은 확률 구간은 비교적 조용하다.
- 나쁜 그림: 모든 구간에서 확률이 비슷하거나, 실제 움직임과 반대로 간다.

### 7.3 Decision mask graph

- 데이터: point branch와 risk branch의 test 구간 정렬 결과
- x축: test time index
- y축: position 여부
- 목적: point-only 진입과 risk-gated 진입이 얼마나 달라지는지 확인
- 좋은 그림: 위험확률이 높은 구간에서 point-only 진입이 줄어든다.
- 나쁜 그림: 거의 모든 진입을 막아 거래가 사라지거나, 위험 구간을 전혀 막지 못한다.

### 7.4 Policy cumulative return proxy

- 데이터: 실제 다음 수익률과 비용 차감 position PnL
- x축: test time index
- y축: 거래비용 반영 누적수익률 proxy
- 목적: point-only, risk-only, point+risk gate 정책 비교
- 좋은 그림: point+risk gate가 point-only보다 MDD가 작고 누적수익률도 크게 훼손되지 않는다.
- 나쁜 그림: risk gate가 수익 구간까지 모두 막아 누적수익률이 사라진다.

### 7.5 MDD by policy

- 데이터: 각 정책의 equity curve
- x축: MDD
- y축: 정책명
- 목적: 이 프로젝트의 핵심인 하방 방어가 실제로 개선되는지 확인
- 좋은 그림: point+risk gate의 MDD가 point-only보다 덜 음수다.
- 나쁜 그림: MDD는 좋아졌지만 active share가 거의 0이라 사실상 거래를 안 한 결과다.

## 8. 12번에서 기대하는 실현 가능성

12번은 “모든 문제를 해결할 최종 모델”이 아니라 “다음 데이터마트 설계에 필요한 우선순위를 정하는 실험”이다. 따라서 실현 가능성은 높다.

이유는 다음과 같다.

- 10번 학습 엔진을 그대로 재사용한다.
- 11번 risk event 엔진을 그대로 재사용한다.
- 새로 늘리는 것은 모델이 아니라 feature group이다.
- 서버 4090 환경에서 12번만 수행할 예정이므로 case 수를 이전보다 넓게 가져갈 수 있다.
- 결과가 좋지 않아도 어떤 feature group을 버릴지 결정할 수 있어 연구 가치가 있다.

## 9. 12번 실행 산출물

추가된 파일은 다음이다.

- `test/models/12_feature_guardrail_fusion_test.ipynb`
- `test/models/12_feature_guardrail_fusion_test.py`
- `test/experiment_specs/12_feature_guardrail_fusion_plan_20260624.md`

노트북 기본 셀은 다음을 실행한다.

```python
%run test/models/12_feature_guardrail_fusion_test.py --suite feature_guardrail_matrix
```

서버 터미널에서 빠르게 구조만 확인할 때는 다음을 쓴다.

```bash
python test/models/12_feature_guardrail_fusion_test.py \
  --suite feature_ablation_quick \
  --epochs 3 \
  --max-windows 1024 \
  --max-cases 12 \
  --continue-on-failure
```

## 10. 다음 스텝

12번을 실행한 뒤에는 다음 기준으로 결정한다.

1. `coin_liquidity_micro`가 좋아지면 거래대금/유동성 mart를 먼저 정식화한다.
2. `coin_volatility_regime`가 좋아지면 위험 event mart와 regime label을 먼저 만든다.
3. `coin_momentum_reversal`가 좋아지면 기술지표 mart를 정식화한다.
4. `coin_text_context`가 좋아지면 텍스트 mart를 가격 feature pipeline에 본격 결합한다.
5. `coin_full_available`만 좋고 개별 group은 약하면 feature selection이나 regularization을 추가한다.
6. 모든 group이 약하면 feature보다 target horizon, label, decision policy를 다시 본다.

이렇게 해야 데이터마트를 먼저 크게 만들고 나중에 쓸모를 확인하는 순서가 아니라, **실제 최적화 결과가 좋아지는 변수군을 먼저 찾고 그 변수군을 데이터마트로 승격하는 순서**가 된다.

## 11. 최종 판단

10번은 점예측 branch의 최적화 토대다. 11번은 그 점예측을 막무가내로 믿지 않게 하는 위험 guardrail이다. 12번은 이 둘을 결합해 코인 독립변수 후보군을 검증하는 단계다.

따라서 지금 다음 단계는 “모델 설계를 또 바꾸는 것”이 아니라, **독립변수 조합을 바꿔도 10번의 안정 objective가 유지되고 11번의 위험 gate가 MDD 방어에 기여하는지 확인하는 것**이다.

## 12. 참고 자료와 12번 반영 방식

이번 전환은 새 논문을 그대로 재현하는 것이 아니라, 코인 예측 논문들이 공통적으로 쓰는 입력군을 현재 저장소의 데이터 구조에 맞춰 선별한 것이다.

| 참고 흐름 | 문헌에서의 핵심 | 12번 반영 |
|---|---|---|
| 다양한 data source 비교 | 기술지표, 온체인, sentiment/search, 전통시장 지수, macro 변수를 분리해 성능 기여를 봐야 한다는 흐름 | `coin_liquidity_micro`, `coin_volatility_regime`, `coin_text_context`, `coin_cross_market`를 분리 |
| macro/micro factor 기반 Bitcoin 예측 | 가격 자체보다 micro factor와 macro factor의 조합이 regime 설명에 도움 | 현재 mart에 있는 OHLCV proxy부터 쓰고 macro/on-chain은 다음 mart 후보로 보류 |
| sentiment/NLP 기반 코인 예측 | 뉴스·SNS sentiment는 가격값 하나보다 regime, 이벤트, 방향 분류에 보조적 | `coin_text_context`는 있으면 붙이고, 없으면 hard error가 아니라 fallback |
| uncertainty quantification | 점예측만 보지 말고 확률, 예측구간, 의사결정 위험을 같이 본다 | 11번 `absolute_move` probability를 guardrail로 사용 |
| order book / liquidity 연구 | 거래량, spread, depth, imbalance 같은 유동성 정보가 단기 변화에 중요 | 실제 order book이 없으므로 range, turnover, value, Amihud-style proxy를 먼저 테스트 |

사용한 참고 링크:

- Forecasting Cryptocurrency Prices Using Deep Learning: https://arxiv.org/html/2311.14759v1
- Decoding Bitcoin: leveraging macro- and micro-factors in time series analysis for price prediction: https://pmc.ncbi.nlm.nih.gov/articles/PMC11419646/
- From On-chain to Macro: Assessing the Importance of Data Source Diversity in Cryptocurrency Forecasting: https://arxiv.org/html/2506.21246
- A survey of deep learning applications in cryptocurrency: https://pmc.ncbi.nlm.nih.gov/articles/PMC10726249/
- Review of deep learning models for crypto price prediction: https://arxiv.org/html/2405.11431v1
- Deep learning for Bitcoin price direction prediction: https://link.springer.com/article/10.1186/s40854-024-00643-1
- Bitcoin Price Prediction: Peer-Reviewed Evidence and Social Media Forecasting Signals: https://arxiv.org/html/2606.00071v1
- A survey on uncertainty quantification in deep learning for financial time series: https://repositori.uji.es/bitstreams/6cbcd3cc-59f7-4492-8858-a0c9e261c0d5/download
