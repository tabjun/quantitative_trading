# 과거 유사 사건/흐름 데이터마트 구축 리서치 보고서

작성일: 2026-06-08  
브랜치 범위: `stock` 전용  
작업 범위: 업비트 KRW 마켓 전체 종목 대상 historical flow data mart 설계 및 코드 구축. 로컬 전체 마트 빌드, 전체 KRW 수집, 대규모 백테스트, 딥러닝 학습, 노트북 실행은 수행하지 않았다.

---

## 0. 결론 요약

과거 데이터를 현재 분석에 반영하려면 단순히 가격 차트 모양만 비교하면 안 된다. 진정한 의미의 과거 유사 사건은 다음 세 조건이 함께 맞아야 한다.

1. **형태 유사성**: normalized return path, drawdown path, volatility path가 유사해야 한다.
2. **시장 상태 유사성**: 변동성, 거래대금, 유동성, RSI, ROC, Bollinger Band Width, Amihud illiquidity 등 독립변수 상태가 유사해야 한다.
3. **상황/원인 유사성**: 큰 변곡점 당시의 감성, 리스크, 매크로, 규제, 유동성 쇼크 등 맥락 변수가 현재 흐름의 원인과 맞아야 한다.

따라서 이번 구현은 `return_path`만 비교하지 않고, `factor_vector`와 `context_vector`를 함께 저장한다. 유사도는 다음 복합 거리로 계산한다.

```text
composite_distance =
  0.50 * shape_distance
+ 0.30 * factor_distance
+ 0.20 * context_distance
```

기본 대상은 `KRW-BTC`가 아니라 업비트 **KRW 마켓 전체 종목**이다. 원천 테이블은 `upbit_krw_candle(timestamp, ticker, open, high, low, close, volume, value)`이며, 전체 KRW 원천 윈도우를 저장하고 기본 검색 인덱스는 거래대금 상위 유동성 종목으로 별도 구성한다.

---

## 1. 구축 데이터마트 설계

### 1.1 원천 테이블

| 테이블 | 역할 |
| :--- | :--- |
| `upbit_krw_candle` | 전체 KRW 마켓 15분봉 원천 캔들. `ticker` 축 필수. |
| `btc_15m_advance` | 개발 fallback 전용. 운영 기준이 아니다. |
| `text_features_15m` | 이미 구축된 텍스트/감성/리스크 컨텍스트. 존재하면 historical flow window에 자동 결합한다. |

### 1.2 신규 테이블

| 테이블 | 역할 |
| :--- | :--- |
| `historical_flow_windows` | ticker별 sliding window metadata 저장. |
| `historical_flow_shape_features` | 가격 모양, 경로 JSON, shape 요약 저장. |
| `historical_flow_factor_features` | 추세/변동성/유동성 요약과 factor vector 저장. |
| `historical_flow_context_features` | 텍스트 충격, topic exposure, context vector 저장. |
| `historical_flow_features` | 위 3개 테이블을 조인한 호환용 view. |
| `historical_flow_neighbors` | 과거 window 간 top-k 유사 흐름 저장. |
| `historical_flow_event_stats` | 유사 사건 이후 1h/4h/24h/3d 수익률, 미래 MDD, 반등 확률 저장. |
| `historical_regime_stats` | 추세, 변동성, 유동성 bucket 기반 regime label 저장. |
| `historical_flow_run_log` | 빌드 파라미터, 원천 row 수, window length, top-k 기록. |

### 1.3 유사도 구성

| 거리 | 구성 요소 | 목적 |
| :--- | :--- | :--- |
| `shape_distance` | normalized return path | 차트 형태 유사성 |
| `factor_distance` | 수익률, 변동성, MDD, trend slope, value/volume z-score, RSI, ROC, BB width, Amihud | 같은 시장 상태인지 확인 |
| `context_distance` | text event count, sentiment, shock z, risk/macro/crypto/regulation/liquidity count | 변곡점의 원인/상황이 맞는지 확인 |
| `composite_distance` | 세 거리의 가중합 | 최종 historical analog ranking |

### 1.4 변곡점 라벨

| 라벨 | 의미 |
| :--- | :--- |
| `context_confirmed_drawdown` | 가격 급락과 리스크/감성/쇼크 맥락이 함께 확인된 하락 사건 |
| `price_only_drawdown` | 가격은 급락했지만 맥락 변수 확인이 부족한 사건 |
| `context_confirmed_breakout` | 상승 돌파와 긍정 감성/맥락이 함께 확인된 사건 |
| `price_only_breakout` | 가격만 돌파한 사건 |
| `volatility_shock` | 변동성 또는 이벤트 쇼크가 큰 사건 |
| `ordinary_flow` | 일반 흐름 |

---

## 2. 논문별 표준 5단계 리뷰

### 2.1 TimeRAG (2024), *BOOSTING LLM Time Series Forecasting via Retrieval-Augmented Generation*

#### [요약] Summary
과거 시계열을 sliding window로 잘라 knowledge base를 만들고, DTW로 유사 sequence를 검색해 예측 성능을 높이는 RAG형 시계열 프레임워크다.

#### [서론] Introduction
LLM 기반 시계열 예측은 훈련 비용이 크고 도메인 전이가 어렵다. 전체 과거 데이터를 모델 파라미터에 모두 학습시키는 대신, 유사 과거 sequence를 검색해 현재 예측에 제공하는 방식이 필요하다.

#### [분석 기법] Methodology
원 시계열을 window length와 step size로 slicing하고, K-means로 대표 sequence knowledge base를 구성한다. 현재 query sequence와 historical sequence 간 유사도는 DTW로 측정한다.

#### [결과] Results
논문은 다양한 benchmark에서 RAG 결합이 원 모델 대비 평균 2.97% 성능 개선을 보였고, 일부 데이터셋에서는 최대 13.12% 개선을 보고했다.

#### [결론 및 설계 결정] Conclusion
프로젝트에서는 TimeRAG의 sequence KB 개념을 DuckDB 데이터마트로 이식한다. 단, 금융에서는 형태만으로 충분하지 않으므로 DTW 기반 `shape_distance`에 독립변수/컨텍스트 거리를 추가한다.  
**[핵심 인용]**: Time series knowledge base와 DTW retrieval은 과거 유사 흐름을 저장하고 현재 흐름에 붙이는 설계 근거가 된다.

### 2.2 Zhang et al. (2025), *Nearest Neighbor Multivariate Time Series Forecasting*

#### [요약] Summary
전체 장기 multivariate time series를 cached datastore로 저장하고, nearest neighbor retrieval로 예측 모델이 전체 과거 데이터의 희소 유사 패턴을 활용하게 하는 방법이다.

#### [서론] Introduction
일반 MTS 모델은 입력 길이 제한 때문에 긴 과거 전체에 흩어진 유사 패턴과 변수 간 희소 상관관계를 놓친다.

#### [분석 기법] Methodology
대규모 cached series datastore를 만들고, representation 기반 nearest neighbor search로 유사 과거 구간을 검색한다. 모델 재학습 없이 retrieval을 붙이는 구조다.

#### [결과] Results
여러 real-world dataset에서 forecasting performance 개선과 interpretability 향상을 보고했다.

#### [결론 및 설계 결정] Conclusion
업비트 KRW 전체 종목에서는 단일 종목의 짧은 history보다 시장 전체 ticker 축의 유사 사건이 중요하다. 따라서 `ticker`, `window_id`, `factor_vector_json`, `context_vector_json`을 저장해 cross-ticker analog를 지원한다.  
**[핵심 인용]**: 전체 데이터셋의 cached series를 검색하는 구조는 “미리 분석해놓은 과거 흐름 마트”의 직접 근거다.

### 2.3 Luan & Hamp (2026), *Automated regime classification in multidimensional time series data using sliced Wasserstein k-means clustering*

#### [요약] Summary
다차원 금융 시계열을 분포 기반으로 clustering하여 market regime을 자동 분류하는 연구다.

#### [서론] Introduction
금융 regime은 단일 가격 방향만으로 설명되지 않고, 수익률 분포, 변동성, 상관구조, 유동성 상태가 함께 바뀐다.

#### [분석 기법] Methodology
Wasserstein k-means를 multidimensional time series에 확장하고, sliced Wasserstein distance로 다차원 분포 차이를 근사한다.

#### [결과] Results
실제 FX spot rate data에서 서로 다른 market regime을 식별할 수 있음을 보였다.

#### [결론 및 설계 결정] Conclusion
프로젝트에서는 경량 구현을 위해 `bull/range/bear`, `low/mid/high_vol`, `low/mid/high_liq` bucket 조합으로 `historical_regime_stats`를 먼저 만든다. 향후 서버 실행 단계에서 Wasserstein 기반 regime clustering으로 확장할 수 있다.  
**[핵심 인용]**: regime은 가격 하나가 아니라 다차원 시계열 분포의 상태로 정의해야 한다.

### 2.4 Dynamic Time Warping for Lead-Lag Relationships (2023)

#### [요약] Summary
DTW를 통해 서로 다른 시계열 간 시간 지연과 lead-lag 관계를 분석하는 방법론이다.

#### [서론] Introduction
금융시장에서는 한 종목이 먼저 움직이고 다른 종목이 뒤따라 움직이는 동조/전이 현상이 자주 발생한다.

#### [분석 기법] Methodology
DTW alignment path를 사용해 두 sequence의 비선형 시간 정렬과 선행/후행 구조를 평가한다.

#### [결과] Results
lead-lag 관계 탐색에서 고정 lag 상관보다 유연한 시간 정렬이 가능함을 보인다.

#### [결론 및 설계 결정] Conclusion
KRW 전체 종목 데이터마트에서 cross-ticker historical analog를 허용한다. 현재 종목과 다른 종목의 과거 흐름도 유사 사건 후보가 될 수 있다.  
**[핵심 인용]**: 유사 사건은 반드시 같은 ticker 안에서만 찾을 필요가 없고, 시장 내 lead-lag 전이 구조를 고려해야 한다.

### 2.5 Matrix Profile 기반 금융 시계열 motif/discord 연구

#### [요약] Summary
Matrix Profile은 시계열 내 반복 motif와 이상 discord를 빠르게 찾는 all-pairs subsequence similarity 기법이다.

#### [서론] Introduction
금융 시계열의 큰 변곡점은 반복되는 motif와 갑작스러운 discord 형태로 나타날 수 있다.

#### [분석 기법] Methodology
subsequence 간 거리 profile을 계산해 가장 가까운 motif와 가장 먼 discord를 식별한다.

#### [결과] Results
금융 시계열 분석에서 motif discovery와 anomaly detection에 활용 가능함이 보고된다.

#### [결론 및 설계 결정] Conclusion
이번 1차 구현은 DuckDB에 window와 vector를 저장하고 L2/DTW 기반 top-k를 만든다. 향후 서버 단계에서 Matrix Profile 기반 근사 인덱스를 추가하면 전체 KRW 마켓의 window 수가 커져도 빠른 검색이 가능하다.  
**[핵심 인용]**: 반복 패턴과 이상 패턴을 사전에 계산해 저장하는 방식은 historical flow mart의 효율화 근거다.

---

## 3. 프로젝트 적용 결정

1. 전체 KRW 마켓 원천 window는 모두 저장한다.
2. 기본 neighbor index는 거래대금 상위 `liquid_top_n=50` 종목으로 구성한다.
3. 유사도는 가격 형태만 쓰지 않고 `shape + factor + context` 복합 거리로 계산한다.
4. 텍스트/감성 컨텍스트가 이미 `text_features_15m`에 있으면 자동 결합하고, 없으면 0으로 채워 가격/거래/기술지표 기반 마트만 구축한다.
5. 대규모 전체 빌드는 학교 서버/자동화 환경에서 실행하고, Codex 로컬 세션에서는 문법 점검과 tiny synthetic test만 수행한다.

## 4. 구축 코드 요약

| 파일 | 역할 |
| :--- | :--- |
| `marts/historical_flow.py` | 전체 KRW 마켓 historical flow mart 생성/조회 핵심 모듈 |
| `pipelines/build_historical_flow_mart.py` | 서버/자동화 환경에서 실행할 빌드 entrypoint |
| `pipelines/query_historical_flows.py` | 현재 ticker 흐름과 유사한 과거 사건 조회 CLI |

운영 실행 예시는 다음과 같다. 이 명령은 전체 KRW 마켓을 대상으로 하므로 로컬 Codex 세션에서는 실행하지 않고, 학교 서버 또는 자동화 환경에서 실행한다.

```bash
uv run pipelines/build_historical_flow_mart.py --window-lengths 16,48,96,288 --stride 4 --top-k 10 --liquid-top-n 50
```

조회 예시는 다음과 같다.

```bash
uv run pipelines/query_historical_flows.py --ticker KRW-SOL --window-length 96 --top-k 10
```

결과 해석은 `query_composite_distance`가 낮을수록 현재 흐름과 과거 사건이 더 유사하다는 뜻이다. 단순 형태 유사성은 `query_dtw_distance`, 독립변수 상태 유사성은 `query_factor_distance`, 당시 원인/맥락 유사성은 `query_context_distance`로 분해해 확인한다.

---

## 5. 참고 링크

| 논문/자료 | 링크 |
| :--- | :--- |
| TimeRAG | https://arxiv.org/abs/2412.16643 |
| Nearest Neighbor Multivariate Time Series Forecasting | https://arxiv.org/abs/2505.11625 |
| Automated regime classification with sliced Wasserstein k-means | https://arxiv.org/abs/2310.01285 |
| Dynamic Time Warping for Lead-Lag Relationships | https://arxiv.org/abs/2309.08800 |
| Matrix Profile 금융 시계열 motif/discord 연구 | https://www.mdpi.com/2673-4591/5/1/45 |
