# 연구 수행 프로세스 (Research Process)

본 문서느 긴 호흡의 AI 에이전트 세션(Codex CLI 등)이 토큰 한계로 인해 망가지는 것을 방지하고, 사용자의 주기적 세션 초기화 요구에 완벽하게 대응하기 위해 작업을 모듈화된 **'단위 프로세스(Step)'**로 나눈 가이드라인입니다.

## 💡 AI 에이전트 프로세스 진행 및 세션 복원 규칙
- **세션당 범위 제한:** 한 세션(대화)에서는 **최대 1~2개의 Step**만 완료하는 것을 목표로 합니다.
- **세션 종료 전 자동 기록:** 각 Step이 완료되면 세션을 종료하기 전에 반드시 프로젝트 루트의 `history.md`와 `process.md`에 수행 결과와 다음 Step을 명확히 기록(Update)해야 합니다.
- **세션 초기화 후 즉각 복원:** 새로운 세션이 초기화되고 복원되어 작업을 다시 수행할 때, `Codex CLI` 에이전트는 `AGENTS.md`에 의거하여 **가장 먼저 `history.md`와 `process.md`를 읽고** 복원 지점(ToDo 리스트)으로 정확하게 진입하여 작업을 이어서 수행해야 합니다.

---

## Phase 1: 데이터 마트(Data Mart) 기반 구축
목표: 매번 API를 호출하는 낭비를 줄이고, 다변량 분석이 가능하도록 과거 데이터를 정제하여 자체 DB화합니다.

- **Step 1.1: 시장 데이터 수집기 연동**
  - Upbit API, 증권사 MTS 연동을 통해 과거 시계열 데이터(15분/일봉 등)를 대량 수집합니다.
- **Step 1.2: 매크로/감성 데이터 수집**
  - 과거 주요 사건(전쟁, 금리 발표 등) 당시의 뉴스 기사, SNS 반응 등 비정형 텍스트 데이터를 수집 및 임베딩 벡터로 변환합니다.
- **Step 1.3: Data Mart 스키마 설계 및 적재**
  - SQLite, DuckDB, AWS RDS 등의 DB를 구축하여 위 수집 데이터를 '섹터별', '타임라인별'로 빠르게 조회할 수 있도록 구조화합니다.

## Phase 2: 다변량 분석 엔진 개발 (Multivariate Engine)
목표: 단일 시계열(가격) 분석을 넘어, 과거 유사 패턴과 당시의 감성(Context)을 융합합니다.

- **Step 2.0: 기초 시계열 모델 성능 비교 (Completed)**
  - RNN, LSTM, GRU, Transformer, ODE-RNN 등 주요 알고리즘의 성능을 업비트 데이터를 통해 벤치마킹합니다.
  - **(Advanced) 고도화된 단변량 분석 (2026-05-19):** 
    - DuckDB(`upbit_data.db`) 기반 대규모 시계열 데이터(3년, 15분 단위) 관리 체계 구축.
    - TCN, N-BEATS Style, XGBoost 등 딥러닝과 머신러닝을 결합한 하이브리드 알고리즘 스택 확장.
    - 검증 손실 기반 가중 앙상블 기법 적용.
  - **(Advanced) 예측 지연 극복 및 통계 진단 고도화 (2026-05-24 - Completed):**
    - 5종 시계열 전처리 파이프라인(`PreprocessingPipeline`) 구축 (차분, 로그수익률, Z-score 전처리)으로 지연 예측(Lagging) 현상 전면 극복.
    - 1 에포크 만에 수직낙하 하던 훈련 손실의 identity mapping 원인 규명 및 지연극복 전처리로 점진적 하강 곡선 안정화.
    - 엄격한 시간 기반 2년 학습 / 1년 테스트 strict split 설계 구현.
    - Durbin-Watson, Jarque-Bera, Ljung-Box 통계 검정을 융합한 잔차 진단 한글 자동 리포터 탑재 및 로컬 5종 300 DPI 이미지 저장 연동.
  - **(Advanced) 업비트 다중 종목(Multi-Ticker) 변동성 추적 및 고속 시계열 예측 (2026-05-25 - Completed):**
    - 업비트 전체 250여 개 종목 중 변동성 및 거래대금(score = volatility * log(value))이 극대화된 상위 종목을 동적으로 색출하는 Dynamic Volatility Ticker Filter 구현.
    - 2단계 분석 결과(MinMax-Autoformer 우수성 실증)에 기초하여 스케일 격차 극복을 위한 MinMaxScaler 단일 공식 채택 및 논문 근거 기술 확보.
    - 데이터 스플릿을 1년 학습 / 3개월 테스트 단일 Hold-out으로 축소하여 훈련 Epoch를 1회로 극소화한 고속 시계열 학습 및 예측 아키텍처 완성.
    - 최종 3개월 아웃오브샘플 예측 캔들차트(`test/images/3_multi_ticker_forecast.png`) 및 실험 프레임워크 명세서(`test/experiment_specs/3_time_series_multi_ticker_test_framework_design_20260525_214347.md`) 생성 완료.
- **Step 2.1: 시계열 패턴 매칭 (DTW 구현)**
  - `fastdtw` 등을 활용하여 현재 주가 하락 파동과 가장 유사한 과거 폭락장 사례를 추출하는 모듈을 개발합니다.
- **Step 2.2: 상황 및 감성 분석 (NLP Context)**
  - 과거 폭락 당시의 텍스트 임베딩과 현재 이슈 텍스트 임베딩 간의 코사인 유사도(Cosine Similarity)를 산출합니다.
- **Step 2.3: 다변량 통합 로직 결합**
  - 차트 형태(DTW)와 시장 감성(NLP) 유사도를 가중 평균하여 최종 '역사적 유사 국면'을 특정하는 핵심 로직을 완성합니다.

## Phase 3: "최하방 방어" 백테스트 (Validation)
목표: 교수님의 핵심 요구사항인 '절대 잃지 않는' 자금 관리 규칙을 과거 데이터에 시뮬레이션합니다.

- **Step 3.1: 퀀트 방어 로직 코딩 (Completed - 2026-05-28)**
  - MDD를 최소화하기 위해 '30% 현금 유동성 의무 확보', '매수가 대비 기계적 -2% 손절선(Stop-loss)', '단기 이동평균선(SMA-5/SMA-20) 돌파 및 거래량 동반(Breakout) 시 진입, 이평선 하회 시 철저히 관망' 등의 방어적 매매 전략을 설계하고, **모의 투자 2차 업그레이드(_v2)를 통해 업비트 표준 거래수수료 0.05% 및 호가 스프레드/체결지연 슬리피지 0.02%를 완벽하게 통합 산출하는 거래 비용 모델**을 코딩하여 포트폴리오 자산 마찰을 실전 퀀트 수준으로 극대화하여 구현했습니다.
- **Step 3.2: Data Mart 연동 백테스트 (Completed - 2026-05-28)**
  - DuckDB `upbit_data.db`의 3년치 10.4만 행 고빈도 `btc_15m_advance` 테이블을 연동하여, pandas/numpy 기반으로 매틱마다 실시간 SMA 지표 및 국소 지지/저항선을 계산하는 모의 투자 시뮬레이터 v2를 기동하였습니다.
  - 단순 코드 구동을 넘어 **Codex 내장 인지 모델 자체가 실시간 캔들 모양(Doji, Hammer, Marubozu, 상승장악형, 유성선)을 직접 시각적으로 판독**하고 거래를 기계적으로 집행하며 거래비용(수수료+슬리피지) 마찰의 가치를 극적으로 실증해 낸 보고서(`analysis_report.md`)를 생성하여 교수님께 메일 전송을 완료했습니다.
- **Step 3.3: 하이퍼파라미터 튜닝**
  - 변동성 돌파 계수($k$), 손절 라인 등 파라미터를 조정하며 가장 안정적인 기대값을 도출합니다.

## Phase 4: 시각화 및 리포팅 (Presentation)
목표: 분석된 다변량 데이터를 직관적으로 이해할 수 있는 결과물로 도출합니다.

- **Step 4.1: 결과 시각화 모듈 (Dashboard)**
  - 차트 매칭 결과 및 자산 곡선을 시각화하는 모듈을 개발합니다.
- **Step 4.2: AI 상황 복기 리포트 생성기**
  - "왜 떨어지고 있으며, 과거 유사 사례에서 반등의 트리거는 무엇이었는가"를 설명하는 논리적 텍스트 리포트를 LLM을 통해 자동 생성합니다.
## MCP Recovery Addendum

- [x] Retarget the arXiv MCP config to `arxiv-mcp-server` in `.agents/mcp_config.json`, `.codex/config.toml`, and `.mcp.json`.
- [x] Update `README.md` and `AGENTS.md` so the documented launch path matches the installed server binary.
- [ ] Verify the Codex MCP loader picks up `arxiv-mcp-server`, then keep that as the next checkpoint.

## 2026-06-07 Session Addendum

- [x] Reconciled the migration docs by rewriting the test docs to match the current Codex / `arxiv-mcp-server` workflow.
- [x] Cleaned obvious generated clutter under `test/` and kept only the useful experiment artifacts.
- [ ] Re-run a quick file-tree review if more temporary outputs appear during the next research pass.

## 2026-06-07 Temporary Mail Prep Addendum

- [x] Generated the temporary Upbit spot backtest report in `quantitative_trading/AI_trading_temporary/analysis_report.md`.
- [x] Committed and pushed the report to `origin/stock` so the GitHub link is available for email delivery.
- [x] Dispatch the email with `test/scripts/send_email.py` after loading `test/.env`.
- [x] Remove the temporary attachment copy under `test/` after dispatch.
- [x] Replace the mojibake email template with a clean UTF-8 Korean link-only delivery message.
- [x] Remove the old temporary report and resend the corrected email with `AI_trading_youtube_upbit_report_20260607.md`.
- [x] Restore `test/scripts/send_email.py` after the resend so the only remaining temporary AI trading artifact is the committed markdown report.

## 2026-06-08 Realtime Text Context Addendum

- [x] Added `text_context.py` for realtime text collection from RSS, optional Naver News API, and local report/SNS CSV exports.
- [x] Added DuckDB mart tables `text_events_raw` and `text_features_15m` with 15-minute independent variables: event count, sentiment mean/sum, shock z-score, 1-hour sentiment momentum, and macro/risk/crypto/regulation/liquidity topic counts.
- [x] Added `ingest_text_context.py` as the operational entrypoint: `uv run ingest_text_context.py`.
- [x] Connected `simulate_and_send.py` to text factors and added `text_risk_guard` so strongly negative/risk-heavy text context can block new entries.
- [x] Verified ingestion end-to-end: 90 RSS records refreshed, 91 total raw text rows retained, and 500 candle-aligned feature rows generated.
- [ ] Next checkpoint: refresh or extend the realtime Upbit candle mart so 2026-06-08 text timestamps overlap price buckets, then run a text-aware backtest/report pass without triggering email delivery.
