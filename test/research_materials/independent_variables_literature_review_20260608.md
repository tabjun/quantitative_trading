# 주식/코인 예측 독립변수 설계 근거 보고서

작성일: 2026-06-08  
브랜치 범위: `stock` 전용  
작업 범위: 논문 기반 독립변수 설계, 문서화, 메일 전달. 로컬 대용량 학습/백테스트/노트북 실행 없음.

---

## 0. 결론 요약

주식과 코인 예측에서 단순 OHLCV만 사용하는 것은 비정상성, 레짐 전환, 지연 매핑, 수수료 마찰을 충분히 설명하지 못한다. 최근 논문들은 가격/거래량 외에 다음 독립변수를 결합할 때 예측 안정성, 해석 가능성, 위기 구간 방어력이 개선된다고 보고한다.

1. 가격/수익률/변동성 계열: log return, realized volatility, high-low range, GARCH/EGARCH proxy.
2. 기술지표 계열: RSI, ROC, Bollinger Band Width, moving-average spread, breakout distance.
3. 거래량/유동성 계열: volume z-score, value traded, Amihud illiquidity, turnover, spread proxy.
4. 오더북/미시구조 계열: bid-ask spread, order imbalance, depth imbalance, multi-level LOB volume profile.
5. 뉴스/리포트 감성 계열: FinBERT/DeBERTa/RoBERTa 또는 LLM ensemble sentiment, sentiment count/sum/max/min.
6. SNS/커뮤니티 계열: topic sentiment, post/tweet volume, verified/follower-weighted sentiment.
7. 한국어 금융 텍스트 계열: entity-level/aspect-level sentiment, TGT-Masking style entity leakage control.
8. 매크로/레짐 계열: CPI, unemployment, yield spread, GDP growth, interest rate, FX, inflation surprise.
9. 글로벌 유동성/크립토 매크로 계열: Global M2 liquidity, lagged liquidity, dollar index, ETF flow proxy.
10. 온체인/거래소 계열: exchange inflow/outflow, active addresses, funding rate, open interest, liquidation volume.
11. 크로스마켓 계열: NASDAQ/S&P500 returns, VIX, DXY, 금리, 금, 원화 환율, BTC dominance.
12. 이벤트/쇼크 계열: text_event_count, text_shock_z, macro_event_dummy, regime similarity score.

설계 우선순위는 다음과 같다.

| 우선순위 | 독립변수 그룹 | 이유 | 주파수/정렬 |
| :--- | :--- | :--- | :--- |
| 1 | OHLCV + 변동성 + 기술지표 | 모든 자산에 공통이며 baseline drift를 통제 | 15분봉 기준 rolling |
| 2 | 뉴스/리포트/SNS 감성 | 가격이 반응하기 전 narrative shock을 포착 | 발행시각을 15분 bucket으로 floor |
| 3 | 매크로/레짐 | 분포 이동과 구조적 시장 전환을 설명 | 일/월 단위 forward-fill, lag feature |
| 4 | 코인 유동성/온체인 | 코인 특유의 자금 흐름과 레버리지 압력을 설명 | 시간/일 단위, lag/rolling |
| 5 | 오더북/스프레드 | 단기 체결 가능성과 수수료 마찰을 직접 반영 | tick/초/분 단위 집계 |

---

## 1. 독립변수 설계안

### 1.1 가격·수익률·변동성 변수

| 변수명 | 정의 | 변환 | 근거 |
| :--- | :--- | :--- | :--- |
| `log_return_1` | `log(close_t / close_t-1)` | z-score 또는 robust scaling | 비정상 가격 수준 대신 변화율 학습 |
| `return_4`, `return_16`, `return_96` | 1시간/4시간/1일 수익률 | rolling window | 단기/중기 momentum 분리 |
| `realized_vol_16` | 15분 수익률의 4시간 표준편차 | rolling | 변동성 군집 포착 |
| `hl_range_pct` | `(high-low)/close` | winsorize | 장중 불확실성 proxy |
| `gk_volatility` | Garman-Klass 변동성 | rolling | 고가/저가/시가/종가 정보 활용 |

### 1.2 기술지표 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `rsi_14` | 14-period RSI | Kadiyala & Mirzaeinia(2025)가 LLM 감성과 RSI/ROC/BBW를 함께 사용 |
| `roc_12` | Rate of Change | 단기 momentum |
| `bb_width_20` | Bollinger Band Width | 변동성 확장/수축 상태 |
| `sma_spread_5_20` | `(SMA5-SMA20)/close` | 추세 전환/골든크로스 |
| `breakout_distance` | `(close - rolling_high)/close` | 돌파 강도 |

### 1.3 거래량·유동성 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `volume_z_96` | 1일 rolling volume z-score | 거래량 shock |
| `value_z_96` | 거래대금 z-score | 실제 자금 유입 강도 |
| `amihud_illiq` | `abs(return)/value` | 가격 충격 대비 유동성 |
| `turnover_proxy` | `volume / rolling_mean(volume)` | 시장 관심도 |
| `spread_proxy` | 고빈도 호가 부재 시 `(high-low)/close` | 거래 마찰 proxy |

### 1.4 뉴스·리포트·SNS 텍스트 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `text_event_count` | 15분 bucket 내 텍스트 수 | Siala et al.(2026)에서 news count ablation 중요성 확인 |
| `text_sentiment_mean` | 감성 평균 | 뉴스 방향성 |
| `text_sentiment_sum` | 감성 합 | 같은 방향 뉴스가 몰릴 때 강도 반영 |
| `text_sentiment_min/max` | 가장 비관/낙관적 신호 | 꼬리 이벤트 반영 |
| `text_sentiment_majority` | 다수결 감성 | noise 완화 |
| `topic_sentiment_k` | BERTopic topic별 감성 | Zhu & Yen(2024)의 topic-level sentiment 근거 |
| `text_shock_z` | 텍스트 이벤트 수의 rolling z-score | 이벤트 burst 감지 |
| `source_weighted_sentiment` | 신뢰도/팔로워/인증 여부 가중 감성 | Haritha & Sahana(2023)의 user following/verified 변수 근거 |

### 1.5 한국어 금융 텍스트 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `kor_entity_sentiment` | 특정 종목/코인명에 연결된 entity-level sentiment | Son et al.(2023)의 KorFinASC |
| `kor_aspect_sentiment` | 실적, 규제, 수급, 금리 등 aspect별 감성 | 문장 전체 감성보다 세밀한 원인 분해 |
| `entity_masked_sentiment` | 종목명 지식 누수 억제 후 감성 | TGT-Masking은 비정상 entity knowledge 억제 |

### 1.6 매크로·레짐 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `cpi_yoy`, `inflation_surprise` | 물가/예상 대비 차이 | macro-contextual retrieval, HANET |
| `unemployment_rate` | 실업률 | 경기 레짐 |
| `yield_spread_10y_2y` | 장단기 금리차 | 경기침체/위험선호 |
| `policy_rate` | 기준금리 | 유동성/할인율 |
| `gdp_growth` | 성장률 | 장기 레짐 |
| `macro_regime_similarity` | 현재 macro vector와 과거 구간 cosine/DTW similarity | Khanna et al.(2025)의 macro-contextual retrieval |

### 1.7 코인 특화 유동성·온체인 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `global_m2_lag_12w` | 18개국 Global M2의 12주 lag | Karthick(2025)의 Bitcoin Global M2 liquidity |
| `exchange_inflow/outflow` | 거래소 입출금량 | 매도 압력/축적 |
| `active_addresses` | 활성 주소 | 네트워크 사용도 |
| `funding_rate` | 무기한 선물 funding | 레버리지 포지셔닝 |
| `open_interest` | 미결제약정 | 청산 위험 |
| `liquidation_volume` | 강제청산 규모 | 레버리지 shock |

### 1.8 오더북·미시구조 변수

| 변수명 | 정의 | 근거 |
| :--- | :--- | :--- |
| `bid_ask_spread` | 최우선 매수/매도 호가 차이 | TLOB, Deep LOB forecasting |
| `order_imbalance_l1` | `(bid_size-ask_size)/(bid_size+ask_size)` | 단기 수급 불균형 |
| `depth_imbalance_k` | k-level 누적 depth imbalance | 공간적 LOB 구조 |
| `lob_volume_profile` | 호가 레벨별 volume vector | TLOB dual attention |
| `spread_adjusted_label` | 평균 spread를 반영한 상승/하락 라벨 | TLOB이 거래비용 반영 필요성 강조 |

---

## 2. 최종 Data Mart 스키마 제안

```text
market_features_15m
- timestamp
- ticker
- open, high, low, close, volume, value
- log_return_1, return_4, return_16, return_96
- realized_vol_16, hl_range_pct, gk_volatility
- rsi_14, roc_12, bb_width_20, sma_spread_5_20, breakout_distance
- volume_z_96, value_z_96, amihud_illiq, turnover_proxy, spread_proxy

text_features_15m
- timestamp
- ticker_hint
- text_event_count
- text_sentiment_mean, text_sentiment_sum, text_sentiment_min, text_sentiment_max
- text_sentiment_majority
- text_shock_z
- topic_sentiment_macro, topic_sentiment_regulation, topic_sentiment_liquidity, topic_sentiment_risk
- kor_entity_sentiment, kor_aspect_sentiment, entity_masked_sentiment

macro_features_daily_monthly
- date
- cpi_yoy, inflation_surprise, unemployment_rate, yield_spread_10y_2y
- policy_rate, gdp_growth, dxy, vix, nasdaq_return, sp500_return
- macro_regime_similarity

crypto_features_hourly_daily
- timestamp
- ticker
- global_m2_lag_12w
- exchange_inflow, exchange_outflow
- active_addresses
- funding_rate, open_interest, liquidation_volume
- btc_dominance

lob_features_intraday
- timestamp
- ticker
- bid_ask_spread
- order_imbalance_l1
- depth_imbalance_5
- lob_volume_profile_json
- spread_adjusted_label
```

정렬 원칙:

- 가격 기준은 15분봉 `timestamp`.
- 뉴스/SNS는 발행시각을 15분 단위로 floor.
- 일/월 매크로 변수는 발표 지연과 look-ahead bias를 막기 위해 실제 발표 가능 시점 이후 forward-fill.
- 코인 유동성/온체인 변수는 API 제공 주파수에 맞춰 lag/rolling 적용.
- 오더북 변수는 가능하면 별도 고빈도 table로 보존하고 15분봉에는 summary statistic만 join.

---

## 3. 논문별 표준 5단계 리뷰

### 3.1 Siala, Khanfir, Papadakis (2026), *Impact of LLMs news Sentiment Analysis on Stock Price Movement Prediction*

- 링크: https://arxiv.org/abs/2602.00086

#### [요약] Summary
FinBERT, RoBERTa, DeBERTa 및 ensemble sentiment를 주가 방향 예측 모델에 결합하면 일부 분류/회귀 모델에서 성능이 개선되며, 특히 news count와 sentiment sum 계열이 유의미하다.

#### [서론] Introduction
기존 연구는 감성분석 모델 평가와 주가 예측 모델 평가를 분리해 다루는 경향이 있었다. 이 논문은 뉴스 감성이 실제 주가 movement prediction에서 어떤 표현 방식으로 도움이 되는지 비교한다.

#### [분석 기법] Methodology
MSFT, AMZN, AAPL, NFLX, TSLA에 대해 2022-03-10부터 2025-04-02까지 주가와 약 96,000건의 뉴스 데이터를 사용한다. FinBERT, RoBERTa, DeBERTa의 감성 출력과 LR/RF/SVM ensemble을 만들고, LSTM, PatchTST, TimesNet, tPatchGNN과 결합한다. 일별 aggregation은 감성 score sum, min, max, majority vote, label별 count를 포함한다.

#### [결과] Results
DeBERTa는 감성 분류 정확도 약 75%를 기록했고, SVM ensemble은 약 79% 수준의 accuracy/F1을 보였다. LSTM ablation에서 전체 감성 feature 사용 시 AUC 0.5557, F1 0.5507이었고, count/sum 제거 시 성능이 하락했다.

#### [결론 및 설계 결정] Conclusion
우리 프로젝트는 `text_event_count`, `text_sentiment_sum`, `text_sentiment_mean`, `text_sentiment_min/max`, `text_sentiment_majority`를 독립변수로 채택한다. 특히 단순 평균만 쓰지 말고 count와 sum을 같이 보존해야 한다.

**[핵심 인용]** 논문은 daily sentiment aggregation에서 count와 sum feature 제거가 예측 성능을 낮춘다고 보고한다.

---

### 3.2 Zhu & Yen (2024), *BERTopic-Driven Stock Market Predictions: Unraveling Sentiment Insights*

- 링크: https://arxiv.org/abs/2404.02053

#### [요약] Summary
온라인 주식 댓글을 BERTopic으로 topic cluster화한 뒤 topic-level sentiment를 딥러닝 시계열 모델에 결합하면 가격 추세와 변동성 이해에 도움이 된다.

#### [서론] Introduction
개별 문장 단위 감성은 noise가 크고 주제 맥락을 잃기 쉽다. 투자자 댓글과 SNS는 정치, 경제, 심리, 기업 이슈가 섞여 있어 topic별 분해가 필요하다.

#### [분석 기법] Methodology
BERT embedding, UMAP 차원축소, HDBSCAN clustering을 사용하는 BERTopic으로 주식 관련 댓글을 topic별로 묶고, 각 topic의 bullish/bearish sentiment를 추출한다. 이후 여러 딥러닝 모델의 price prediction에 topic feature 유무를 비교한다.

#### [결과] Results
논문은 topic sentiment를 추가한 모델이 기존 모델 대비 더 나은 예측력을 보이며, stock market comments의 topic이 volatility와 price trend에 implicit signal을 제공한다고 보고한다.

#### [결론 및 설계 결정] Conclusion
우리 프로젝트는 `topic_sentiment_macro`, `topic_sentiment_regulation`, `topic_sentiment_liquidity`, `topic_sentiment_risk`처럼 topic별 감성을 분리한다. 모든 텍스트를 하나의 sentiment 평균으로 합치지 않는다.

**[핵심 인용]** 논문은 topic-level stock comment sentiment가 가격 추세와 변동성에 대한 암묵적 정보를 제공한다고 해석한다.

---

### 3.3 Son et al. (2023), *Removing Non-Stationary Knowledge From Pre-Trained Language Models for Entity-Level Sentiment Classification in Finance*

- 링크: https://arxiv.org/abs/2301.03136

#### [요약] Summary
한국어 금융 텍스트에서는 entity-level/aspect-level sentiment가 중요하며, 사전학습 모델이 가진 오래된 종목 지식이 예측 성능을 과대평가할 수 있어 TGT-Masking으로 억제해야 한다.

#### [서론] Introduction
한국어 금융 텍스트는 영어 대비 finance-specific annotation이 부족하다. 종목명, 기업명, 코인명에 대한 과거 지식은 시간이 지나면 틀릴 수 있어 비정상 지식 누수 문제가 생긴다.

#### [분석 기법] Methodology
KorFinASC라는 12,613개 human-annotated Korean aspect-level sentiment dataset을 구축하고, intermediate transfer learning과 TGT-Masking을 적용한다. TGT-Masking은 target entity를 직접 추측하지 못하게 하여 PLM의 오래된 entity 지식 사용을 제한한다.

#### [결과] Results
논문은 TGT-Masking을 적용한 transfer learning이 KorFinASC에서 standalone model 대비 classification accuracy를 22.63% 개선했다고 보고한다.

#### [결론 및 설계 결정] Conclusion
국내 주식/원화 코인 분석에서는 `kor_entity_sentiment`, `kor_aspect_sentiment`, `entity_masked_sentiment`를 별도 변수로 둔다. 종목명 자체를 감성 모델이 기억해 만든 bias를 줄이기 위해 entity masking 기반 feature를 고려한다.

**[핵심 인용]** 논문은 finance PLM의 non-stationary entity knowledge가 predictive power를 과대평가할 수 있다고 지적한다.

---

### 3.4 Khanna et al. (2025), *History Rhymes: Macro-Contextual Retrieval for Robust Financial Forecasting*

- 링크: https://arxiv.org/abs/2511.09754

#### [요약] Summary
macro indicators와 financial news sentiment를 공동 embedding하여 과거 유사 macro regime을 검색하면 OOD regime에서 더 견고하고 해석 가능한 예측이 가능하다.

#### [서론] Introduction
금융시장은 구조적 변화와 macro regime shift 때문에 cross-validation에서는 좋아도 실제 OOD 시점에서 무너질 수 있다. 단순 numeric/text fusion은 regime shift에 적응하기 어렵다.

#### [분석 기법] Methodology
2007-2023 S&P500 데이터를 사용해 CPI, unemployment, yield spread, GDP growth와 financial news sentiment를 shared similarity space에 embedding한다. inference 시 현재 macro context와 유사한 과거 구간을 retrieval하여 예측 조건으로 사용한다.

#### [결과] Results
OOD 2024 AAPL/XOM 평가에서 macro-conditioned retrieval은 AAPL PF 1.18, Sharpe 0.95, XOM PF 1.16, Sharpe 0.61을 기록하며 static numeric, text-only, naive multimodal baseline보다 견고했다.

#### [결론 및 설계 결정] Conclusion
우리 프로젝트는 `macro_regime_similarity`를 핵심 독립변수로 추가한다. 단순 macro value를 넣는 것보다 현재 구간과 과거 구간의 macro-context 유사도를 함께 모델에 제공해야 한다.

**[핵심 인용]** 논문은 금융 예측을 “현재와 유사한 과거 macro regime을 검색하는 문제”로 재정의한다.

---

### 3.5 Oliveira et al. (2026), *Macro-aware time series forecasting via hierarchical mixed-frequency attention models*

- 링크: https://arxiv.org/abs/2606.00624

#### [요약] Summary
monthly macro context와 daily market return을 hierarchical mixed-frequency attention으로 결합하면 turbulent period에서 risk-adjusted return과 손실 방어가 개선된다.

#### [서론] Introduction
macro-financial 예측에서는 distinct macro regime 수가 적고, 월별 macro와 일별/분봉 market data의 주파수가 다르다. 단순 feature augmentation은 이 mixed-frequency 구조를 제대로 다루지 못한다.

#### [분석 기법] Methodology
HANET은 monthly macro grid에서 query/key attention을 계산하고, daily market representation에 value를 투영한다. 55개 liquid futures, 여러 asset class에 대해 macro-conditioned attention의 효과를 평가한다.

#### [결과] Results
논문은 macro 정보를 무시한 neural forecaster보다 HANET이 특히 turbulent period에서 risk-adjusted returns를 개선하고 손실을 줄였다고 보고한다. Ablation에서는 structured macro conditioning이 naive macro augmentation보다 중요했다.

#### [결론 및 설계 결정] Conclusion
우리 프로젝트는 macro 변수를 15분봉에 무리하게 직접 맞추기보다 `macro_features_daily_monthly` table로 분리하고, 발표 가능 시점 이후 forward-fill 및 lag를 적용한다. macro window 기반 attention/retrieval score를 별도 변수로 둔다.

**[핵심 인용]** 논문은 low-frequency macro signal과 high-frequency return을 보존하는 mixed-frequency attention이 필요하다고 주장한다.

---

### 3.6 Karthick (2025), *Expert System for Bitcoin Forecasting: Integrating Global Liquidity via TimeXer Transformers*

- 링크: https://arxiv.org/abs/2512.22326

#### [요약] Summary
Bitcoin 장기 예측에서 18개국 Global M2 liquidity를 12주 lag exogenous variable로 넣으면 univariate TimeXer 대비 70일 forecast MSE가 크게 개선된다.

#### [서론] Introduction
Bitcoin은 극단적 변동성과 비정상성을 가지며, 단변량 가격 모델은 장기 예측에서 불안정하다. 유동성 환경은 crypto risk appetite과 자금 유입을 설명하는 핵심 외생 변수다.

#### [분석 기법] Methodology
2020-01부터 2025-08까지 Bitcoin daily price에 대해 TimeXer-Exog를 LSTM, N-BEATS, PatchTST, univariate TimeXer와 비교한다. Global M2 liquidity는 18개 주요 경제권을 집계하고 12주 lag 구조로 투입한다.

#### [결과] Results
70일 forecast horizon에서 TimeXer-Exog는 MSE 1.08e8을 기록했고, univariate TimeXer baseline 대비 89% 이상 개선됐다고 보고한다.

#### [결론 및 설계 결정] Conclusion
코인 독립변수에는 `global_m2_lag_12w`를 우선 포함한다. 단기 15분봉 매매라도 중장기 risk-on/off regime을 필터링하는 상위 context 변수로 사용할 수 있다.

**[핵심 인용]** 논문은 Global M2 liquidity가 Bitcoin 장기 예측 안정화에 중요한 leading exogenous variable이라고 제시한다.

---

### 3.7 Berti & Kasneci (2025), *TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction with Limit Order Book Data*

- 링크: https://arxiv.org/abs/2502.15757

#### [요약] Summary
Limit Order Book의 공간적/시간적 구조를 dual attention으로 학습하면 주식과 Bitcoin price trend prediction에서 SOTA 대비 F1 개선이 가능하지만, spread/거래비용 반영 시 성능이 저하된다.

#### [서론] Introduction
LOB는 supply/demand의 가장 세밀한 표현이다. 하지만 market efficiency가 높아질수록 단기 signal은 빠르게 사라지고, 거래비용을 무시하면 예측 성능이 실제 수익으로 이어지지 않는다.

#### [분석 기법] Methodology
TLOB은 LOB price level과 volume level의 spatial dependency, 시간 흐름의 temporal dependency를 dual attention으로 학습한다. FI-2010, NASDAQ Tesla/Intel, Bitcoin dataset에서 여러 horizon을 비교한다.

#### [결과] Results
FI-2010에서 평균 F1 +3.7, Tesla +1.3, Intel +7.7, Bitcoin +1.1 F1 개선을 보고한다. 또한 spread를 반영한 trend labeling에서는 성능이 떨어져 거래비용 고려 필요성을 강조한다.

#### [결론 및 설계 결정] Conclusion
단기 코인/주식 예측에는 `bid_ask_spread`, `order_imbalance_l1`, `depth_imbalance_k`, `lob_volume_profile`, `spread_adjusted_label`을 설계한다. 단, 오더북은 데이터 용량이 크므로 15분봉에는 summary feature만 join한다.

**[핵심 인용]** 논문은 trend prediction을 transaction cost와 함께 평가해야 한다고 지적한다.

---

### 3.8 Haritha & Sahana (2023), *Cryptocurrency Price Prediction using Twitter Sentiment Analysis*

- 링크: https://arxiv.org/abs/2303.09397

#### [요약] Summary
Bitcoin 예측에서 Twitter sentiment, tweet volume, user following, verified 여부를 historical price와 결합하면 GRU 기반 가격 예측에 활용 가능하다.

#### [서론] Introduction
Crypto는 전통 주식보다 SNS와 커뮤니티 narrative에 더 빠르게 반응한다. Twitter는 뉴스 소스이자 투자자 토론 공간으로 기능한다.

#### [분석 기법] Methodology
BERT 기반 sentiment model로 tweet sentiment를 추정하고, GRU로 Bitcoin price를 예측한다. 가격 데이터 외에 tweet volume, user following, verified 여부를 함께 사용한다.

#### [결과] Results
논문은 sentiment prediction MAPE 9.45%, Bitcoin price prediction MAPE 3.6%를 보고한다.

#### [결론 및 설계 결정] Conclusion
코인 텍스트 변수는 감성 점수만 저장하지 말고 `text_event_count`, `source_weighted_sentiment`, `verified_weighted_sentiment`, `influencer_sentiment`를 분리한다.

**[핵심 인용]** 논문은 tweet volume과 사용자 신뢰도 정보를 가격 예측 독립변수로 함께 사용한다.

---

## 4. 프로젝트 적용 결정

### 4.1 즉시 구현 우선순위

1. 기존 `text_features_15m` 확장:
   - `text_sentiment_min`
   - `text_sentiment_max`
   - `text_sentiment_majority`
   - `topic_sentiment_*`
   - `source_weighted_sentiment`
2. 시장 feature builder 추가:
   - `log_return_1`
   - `realized_vol_16`
   - `gk_volatility`
   - `rsi_14`
   - `roc_12`
   - `bb_width_20`
   - `amihud_illiq`
3. macro/crypto table 설계:
   - `macro_features_daily_monthly`
   - `crypto_features_hourly_daily`
4. 오더북은 후순위:
   - Upbit/KRX 호가 API 안정 수집 후 별도 table로 분리.

### 4.2 누수 방지 원칙

- 텍스트 발행시각은 실제 timestamp 기준.
- 장 마감 이후 뉴스는 다음 거래 가능 bucket으로 이동.
- macro 변수는 발표일이 아니라 실제 공개 시각 이후만 사용.
- label은 future return이므로 feature 생성 단계에서 절대 join하지 않음.
- entity sentiment는 종목명 자체 prior를 줄이기 위해 masking variant를 별도로 저장.

### 4.3 모델 입력 권장안

Shallow but Wide 원칙을 유지한다.

- 입력 window: 15분봉 기준 96~384 step.
- feature group embedding:
  - market numerical: 64 dim
  - text context: 32 dim
  - macro/crypto context: 32 dim
  - optional LOB summary: 32 dim
- 모델 depth: 1~2 layers.
- hidden width: 64~128.
- target:
  - regression: next log return 또는 volatility-adjusted return.
  - classification: spread/fee-adjusted directional label.
- 필수 metric:
  - DA
  - MASE
  - fee/slippage-adjusted return
  - MDD
  - regime별 성능 breakdown.

---

## 5. 참고 논문 목록

| arXiv ID | 논문 | 독립변수 근거 |
| :--- | :--- | :--- |
| 2602.00086 | Impact of LLMs news Sentiment Analysis on Stock Price Movement Prediction | LLM news sentiment, count/sum/min/max/majority aggregation |
| 2404.02053 | BERTopic-Driven Stock Market Predictions | topic-level sentiment |
| 2301.03136 | Removing Non-Stationary Knowledge From PLMs for Finance ASC | Korean entity/aspect sentiment, entity masking |
| 2511.09754 | History Rhymes | macro regime similarity, macro+news retrieval |
| 2606.00624 | Macro-aware time series forecasting via HANET | mixed-frequency macro attention |
| 2512.22326 | Bitcoin Forecasting with Global Liquidity | Global M2 liquidity, 12-week lag |
| 2502.15757 | TLOB | LOB spread/depth/order imbalance, spread-adjusted labels |
| 2303.09397 | Cryptocurrency Price Prediction using Twitter Sentiment Analysis | tweet sentiment, tweet volume, follower/verified weighted sentiment |

