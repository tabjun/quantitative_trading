# Text Context Feature Build Report


## 기초통계량(describe 표)

> 2026-10-14 소급 추가(AGENTS.md 2.9.10).

- 해당 없음(이 보고서는 텍스트 피처 생성 기록이며, 가격 테이블 `btc_15m_advance`에서는 최근 타임스탬프 500개만 정렬 기준으로 읽었고 종가·거래량은 쓰지 않았다(`pipelines/ingest_text_context.py`). 당시 범위도 출력이 남아 있지 않아 확인할 수 없다).

---

## Summary
- Built at: 2026-06-08 01:22:37
- DuckDB path: `upbit_data.db`
- Raw text records inserted/refreshed: 90
- 15-minute feature rows generated: 500

## Feature Tables
- `text_events_raw`: normalized news/report/SNS text with lexicon sentiment and topic flags.
- `text_features_15m`: candle-aligned independent variables for multivariate analysis.

## Core Independent Variables
- `text_event_count`: number of text events in the 15-minute bucket.
- `text_sentiment_mean`: average lexicon sentiment, from -1.0 to +1.0.
- `text_shock_z`: rolling 96-bucket z-score for abnormal text volume.
- `text_sentiment_momentum_1h`: 1-hour rolling sentiment mean.
- `text_macro_count`, `text_risk_count`, `text_crypto_count`, `text_regulation_count`, `text_liquidity_count`: topic exposure flags aggregated by bucket.
