# Text Context Feature Build Report

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
