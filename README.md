# AI 퀀트 트레이딩 연구

다변량 시계열 분석, 텍스트 컨텍스트, historical analog data mart를 결합해 하방 방어 중심 퀀트 연구를 수행하는 저장소다.

## 1. 프로젝트 목표

- 단일 가격 시계열이 아니라 다변량 분석으로 시장을 본다.
- 하락장 방어와 MDD 최소화를 우선한다.
- 과거 유사 흐름을 차트 형태뿐 아니라 독립변수와 컨텍스트까지 포함해 찾는다.
- 연구 실험과 실사용 프레임워크를 분리해 관리한다.

## 2. 문서 역할

- `AGENTS.md`: Codex가 매 세션 반드시 지켜야 하는 강제 규칙
- `skills.md`: 기술 철학, 분석 원칙, 보고서 기준
- `history.md`: 지금까지 끝난 작업 이력
- `process.md`: 현재 단계와 다음 작업
- `conversation_l2_cache.md`: 최근 사용자 요청/제약의 압축 캐시

## 3. 디렉터리 구조

- `main.py`: 최상위 진입점
- `analysis/`, `advisors/`, `contexts/`, `integrations/`, `marts/`, `pipelines/`: 실사용 프레임워크 코드
- `data/`: 공용 원천 데이터와 DuckDB 마트 파일
- `database/`: DB 접근 코드와 경로 helper
- `test/models/`: 연구용 노트북과 동명 `.py` 미러
- `test/results/`: 연구 결과물
- `test/research_materials/`: 논문 조사 및 설계 근거
- `test/scripts/`: 재사용 가능한 연구 보조 유틸리티
- `.githooks/`: 저장소 정책 및 노트북 미러 동기화 훅

## 4. 실행 정책

이 저장소에는 자동화 코드와 재현 실행 명령을 남긴다. 다만 Codex는 로컬에서 장시간 연구 실행을 하지 않는다.

- 로컬에서 허용: 코드 작성, 문법 검사, import 확인, 작은 synthetic 테스트
- 로컬에서 금지: 장시간 학습, 백테스트, 노트북 결과 산출
- 실제 연구 실행: 학교 서버 커널, CI, 스케줄러, 사용자 승인 환경

## 5. 연구 워크플로우

1. 세션 시작 시 `AGENTS.md`, `process.md`, `history.md`, `conversation_l2_cache.md`를 읽는다.
2. 기존 실험과 문서를 먼저 검색한다.
3. 연구 실험은 `test/models/*.ipynb`를 먼저 만들고 동명 `.py` 미러를 유지한다.
4. 운영용 기능은 루트 패키지와 `pipelines/`에 구현한다.
5. 결과와 상태를 `history.md`, `process.md`에 반영한다.

## 6. 재현 실행 명령

아래 명령은 학교 서버, CI, 스케줄러, 운영자가 실행할 수 있도록 저장소에 남기는 기준 명령이다.

```bash
uv run main.py
uv run pipelines/ingest_text_context.py
uv run pipelines/build_historical_flow_mart.py
uv run pipelines/query_historical_flows.py
uv run pipelines/simulate_and_send.py
```

## 7. Realtime Text Context Pipeline

뉴스, 시장 리포트, SNS 텍스트를 15분 버킷으로 정렬해 독립변수로 쓰는 파이프라인이다.

```bash
uv run pipelines/ingest_text_context.py
```

- 기본 소스: Google News RSS
- 선택 API: `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`
- 로컬 CSV: `TEXT_LOCAL_CSVS="reports.csv|sns_export.csv"`
- 사용자 RSS: `TEXT_RSS_URLS="https://example.com/rss|https://example.org/feed"`
- DuckDB raw table: `text_events_raw`
- DuckDB feature table: `text_features_15m`

```python
from contexts.text_context import TextFeatureBuilder

builder = TextFeatureBuilder(db_path="data/upbit_data.db")
price_with_text = builder.enrich_price_frame(price_df)
```

## 8. Historical Flow Data Mart

과거 유사 사건/흐름을 Upbit KRW 전체 종목 기준으로 찾는 데이터마트다.

```bash
uv run pipelines/build_historical_flow_mart.py --window-lengths 16,48,96,288 --stride 4 --top-k 10 --liquid-top-n 50
uv run pipelines/query_historical_flows.py --ticker KRW-SOL --window-length 96 --top-k 10
```

- 기본 원천 테이블: `upbit_krw_candle`
- 경량 fallback: `btc_15m_advance`
- DuckDB mart table:
  - `historical_flow_windows`
  - `historical_flow_features`
  - `historical_flow_neighbors`
  - `historical_flow_event_stats`
  - `historical_regime_stats`
  - `historical_flow_run_log`
- 유사도: `query_composite_distance = shape + factor + context`

```python
from marts.historical_flow import HistoricalFlowConfig, HistoricalFlowMart

mart = HistoricalFlowMart(HistoricalFlowConfig(db_path="data/upbit_data.db"))
similar_cases = mart.query_similar_flows(ticker="KRW-BTC", window_length=96, top_k=10)
```

## 9. 메일 및 브랜치 규칙

- 한국어 메일은 UTF-8 기준으로 다룬다.
- 메일 전달 스크립트는 재사용 가능한 형태만 유지한다.
- 사용자가 따로 지시하지 않으면 작업 브랜치에서만 커밋/푸시한다.
- `main`, `develop` 병합은 명시적 요청이 있을 때만 수행한다.
