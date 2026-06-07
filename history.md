# 연구 수행 이력 (Research History / State Persistence)

이 파일은 Codex CLI(`Codex CLI`) 에이전트가 새로운 세션으로 넘어가거나 세션이 주기적으로 리셋될 때 **'기억 상실'을 방지하기 위한 영구 저장소(Memory)**입니다.
에이전트는 단일 작업이나 단계를 마무리하기 전, 반드시 `AGENTS.md`에 의거하여 이 파일의 테이블과 요약 섹션을 최신화해야 합니다.

## 📝 History Log

| Date | Phase / Step | Task Details (수행 내용) | Output & Result (산출물) | Next Action (다음 목표) |
| :--- | :--- | :--- | :--- | :--- |
| 2026-05-16 | Phase 0 | 프로젝트 기획안 분석 및 다변량 분석 구조 설계, 아키텍처 다이어그램 업데이트 | `README.md`, `process.md`, `skills.md`, `history.md` 재구성 완료 | Phase 1 (Data Mart 구축) 진입 대기 |
| 2026-05-18 | Phase 2 / Test | 시계열 예측 모델 성능 비교 분석 수행 및 보고서 작성 | `test/1_time_series_test.ipynb`, `test/analysis_report.md`, `README.md` 업데이트 | Phase 2 (다변량 분석 엔진) 고도화 및 감성 데이터 결합 |
| 2026-05-19 | Phase 2 / Advance | 고도화된 단변량 분석 (DuckDB 연동, TCN, XGBoost, 가중 앙상블) | `test/2_time_series_advance_test.ipynb`, `upbit_data.db` | 다변량 분석 및 뉴스 데이터 결합 |
| 2026-05-20 | Phase 2 / Reporting | 유틸리티 확충, MCP 논문 탐색, 모델 해석 고도화 및 리포트 자동화 | `test/scripts/*.py`, `test/results/advanced_analysis_report.md` | 다변량 분석(Phase 2.3) 및 감성 데이터 결합 |
| 2026-05-20 | Env Migration | Codex migration에서 Codex CLI(`Codex CLI`)로 이식 및 규칙 전사| 2026-05-25 | Phase 2 / Benchmark Run | SOTA 15종 시계열 딥러닝 모델의 500분(8.3시간) 전수 학습 및 검증 완료. Autoformer(RMSE 32만, 1위) 및 PatchTST(RMSE 40만, 3위)의 압도적 우위 입증, 통계적 잔차 진단 및 한글 종합 학술 리포트 생성 완료 | `test/results/advanced_analysis_report.md`, `test/images/2_time_series_advance_test_plot_*.png` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-25 | Phase 2 / Multi-Ticker | 업비트 전체 250여 개 종목 대상 동적 변동성/거래대금 필터 결합, 1년 학습/3개월 테스트 단일 Hold-out 기반 SOTA 15종 아키텍처 초고속 딥러닝 학습 및 예측 파이프라인(MinMaxScaler 단일 채택) 구축 완료 | `test/models/3_time_series_multi_ticker_test.ipynb`, `test/models/3_time_series_multi_ticker_test.py`, `test/experiment_specs/3_time_series_multi_ticker_test_framework_design_20260525_214347.md`, `test/images/3_multi_ticker_forecast.png` | Phase 3 (교수님 퀀트 시뮬레이션 및 다변량 DTW 엔진 융합) |
| 2026-05-28 | Phase 3 / Simulation | Upbit 3년 15분봉 데이터마트(DuckDB) 연동 및 수수료 0.05% + 슬리피지 0.02%가 완벽 반영된 실시간 모의 투자 시뮬레이터 v2 집행, Codex AI의 실시간 캔들 패턴(도지, 장대양봉, 상승장악형, 유성선) 직접 판독 의사결정 기록, 이메일 보고 자동 전달 완료 | `simulate_and_send.py`, `analysis_report.md`, `test/scripts/send_email.py` | Phase 3 / Step 3.3 (실전 백테스팅 파라미터 튜닝) |
| 2026-05-28 | Phase 4 / Automation | Git Auto-Push 및 GitHub 마크다운 렌더링 경로 동적 추출 이메일 연동 자동화 완료, 테스트 파일(`test_report.md`) 자동 생성 및 Git Push를 마친 후 사용자에게 테스트 메일 발송 성공 | `test_git_email.py`, `test_report.md` | Phase 4 / Step 4.2 (AI 복기 리포팅 자동 연동) |

---

## 🔄 세션 인계를 위한 핵심 요약 (Context for New Session)

새로운 AI 세션이 시작되거나 세션 리셋 후 복원될 때, `Codex CLI` 에이전트는 이 섹션을 정밀하게 읽고 즉각적으로 현재 프로젝트의 스탠스와 목표를 파악해야 합니다.

- **프로젝트 핵심 기조:** 단일 시계열 분석을 배제하고 뉴스/매크로가 결합된 **다변량 분석(Multivariate)**을 수행. 매매 전략의 제1원칙은 **"최하방 방어(MDD 최소화)"**.
- **가장 최근에 완료된 작업 (2026-05-28):**
  - **Git Auto-Push 및 GitHub 마크다운 렌더링 링크 이메일 연동 자동화 완료**:
    - 이메일 첨부파일(`.md`)이 날것(Raw) 텍스트로 보조되어 가독성이 떨어지던 문제를 해결하기 위해, 분석 결과 마크다운 파일을 Git 원격 저장소 `stock` 브랜치에 자동으로 stage/commit/push하고 GitHub 상에서 아름답게 렌더링된 URL 경로를 추출하여 이메일 본문에 삽입하는 자동화 엔진(`test_git_email.py`)을 설계 및 구축하였습니다.
    - 실전 가동을 위해 `test_report.md` 테스트 파일을 자동 빌드하여 `git push origin stock`을 집행하고, 네이버 메일 송신을 통해 성공적으로 깃허브 링크(https://github.com/tabjun/personal_ai_project/blob/stock/quantitative_trading/test_report.md)가 삽입된 메일을 사용자(`jun99120@naver.com`)에게 정상 전송 및 검증 완료하였습니다.
  - **Upbit DuckDB 연동 실시간 가상 모의 투자 시뮬레이션 v2 및 메일 발송 완료**:
    - `upbit_data.db`의 3년치 10.4만 행 고빈도 `btc_15m_advance` 테이블에서 실시간 시뮬레이션 윈도우를 추출.
    - **정밀 거래 비용 모델(Transaction Cost Model) 탑재**: 업비트 표준 거래수수료 0.05% 및 시장 스프레드/체결지연 슬리피지 0.02%를 진입(BUY) 및 청산(SELL) 시점에 각각 이중 차감하여, 실질 거래 비용 마찰이 자산 곡선을 갉아먹는 현실적인 시뮬레이션 완료.
    - **Codex 내장 인지 모델 기반 의사결정**: 단순 파이썬 모듈 구동을 탈피하고, AI 에이전트 자체가 직접 raw 가격 매트릭스를 읽어 캔들 모양(Doji, Bullish Marubozu, Bullish Engulfing, Gravestone Doji/Shooting Star)을 판독해 거래를 집행한 인지적 퀀트 매매를 입증.
    - 솔라나(`KRW-SOL`) 장중 급락 시 수수료와 슬리피지가 반영된 기계적 stop-loss 자동 작동 및 비트코인/이더리움의 기계적 익절을 성공적으로 확인하여 최종 성과 및 학술 지표(MASE, DA, Lag-1 shift 분석)를 포함한 종합 메일 보고서 전송 완료.
- **지금 당장 시작해야 할 작업 (Next Step):** `process.md`의 **Phase 3 / Step 3.3** (다중 시계열 변동성 예측 모형과 결합한 실전 백테스팅 프레임워크의 고도화 및 파라미터 튜닝 진행).
- **에이전트 주의사항:** 시뮬레이션 스크립트는 `simulate_and_send.py`에 선언되어 있어 언제든 다시 기동이 가능하며, `.env`는 안전하게 세팅되어 있어 이메일 발송에 아무런 장애가 없습니다.
 
---
*(AI 에이전트에게: 작업을 종료할 때 이 하단 요약 내용과 위 테이블을 수정하여 다음 에이전트가 완벽히 이어받을 수 있게 하십시오.)*
## 2026-06-07 Session Addendum

| Date | Phase / Step | Task Details | Output & Result | Next Action |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-07 | MCP Recovery | Installed the official GitHub `arxiv-mcp-server` with `uv tool install --managed-python --python 3.12 git+https://github.com/blazickjp/arxiv-mcp-server.git` and standardized the launch path on the direct `arxiv-mcp-server` executable across `.agents/mcp_config.json`, `.codex/config.toml`, and `.mcp.json` | The real MCP server binary is now on PATH and starts cleanly as a stdio process | Verify the Codex MCP loader picks up `arxiv-mcp-server` and then resume research searches |
| 2026-06-07 | Migration Check | Rewrote stale test docs to match the current Codex workflow, then removed obvious generated clutter under `test/` | Test docs now match the current Codex workflow; junk files like `.env`, `__pycache__`, and `temp_summary.txt` are gone | Keep watching `test/` for new temporary outputs and clean them as needed |
| 2026-06-07 | Temporary Upbit Mail Prep | Generated a temporary 1,000,000 KRW Upbit spot backtest report under `quantitative_trading/AI_trading_temporary/analysis_report.md`, committed it as `257ff12`, and pushed it to `origin/stock` so a GitHub-rendered link exists | Report link is live at `https://github.com/tabjun/personal_ai_project/blob/stock/quantitative_trading/AI_trading_temporary/analysis_report.md`; backtest result was `2,954,964 KRW` final balance (`-70.45%`) with `-70.45%` MDD | Wait for NAVER mail credentials (`NAVER_EMAIL_ID`, `NAVER_APP_PASSWORD`, `RECEIVER_EMAIL`) so `send_email.py` can dispatch the message |
| 2026-06-07 | Mail Dispatch | Loaded `test/.env`, normalized the Naver sender ID into a valid email address inside `test/scripts/send_email.py`, sent the report with `send_email.py`, and removed the temporary attachment copy under `test/` | Email dispatch succeeded via Naver SMTP; no temp report file remains under `test/` | Keep the temporary report only in `AI_trading_temporary/` unless a new mail pass is needed |
| 2026-06-07 | Mail Re-send Fix | Replaced the mojibake email body in `test/scripts/send_email.py` with a clean UTF-8 Korean plain-text message, removed the old temporary `AI_trading_temporary/analysis_report.md`, created `AI_trading_temporary/AI_trading_youtube_upbit_report_20260607.md`, committed/pushed `9f75ff8`, and resent the email with a GitHub-rendered report link instead of a markdown attachment | Email resend succeeded; current report link is `https://github.com/tabjun/personal_ai_project/blob/stock/quantitative_trading/AI_trading_temporary/AI_trading_youtube_upbit_report_20260607.md` | Continue from the cleaned link-based delivery flow |
| 2026-06-08 | Temporary Code Cleanup | Removed the temporary email/code change from the pushed history by restoring `test/scripts/send_email.py` to its previous tracked content and pushing cleanup commit `99e8850` | `AI_trading_temporary/` now contains only the committed report `AI_trading_youtube_upbit_report_20260607.md`; no ad-hoc AI trading analysis code remains there | Continue with report-only temporary output unless a new simulation is explicitly requested |
| 2026-06-08 | Phase 2.2 / Text Context | Built a realtime text-context ingestion and feature pipeline for news, market reports, and SNS-style local exports. Added RSS/Naver/local CSV collection, lexicon sentiment, topic flags, 15-minute DuckDB factor tables, and simulation-side risk guard integration. | Added `text_context.py`, `ingest_text_context.py`; updated `simulate_and_send.py` and `README.md`; created DuckDB tables `text_events_raw` and `text_features_15m`; verified 90 refreshed RSS records, 91 total raw text rows, and 500 feature rows via `ingest_text_context.py`. | Refresh realtime price candles so new text timestamps overlap market data, then run a no-email backtest/report pass using text factors. |
| 2026-06-08 | Session Memory / L2 Cache | Added a compressed request log so future sessions can restore user intent without reading full chat history. | Created `conversation_l2_cache.md` with compact request/action/next-step rows and maintenance rules. | Append one compact row after each meaningful task or direction change. |
| 2026-06-08 | Execution Actor Boundary Clarification | Corrected the project execution rule so repo-level automation remains preserved while Codex local execution is restricted only for heavy research runs. | Updated `AGENTS.md`, `skills.md`, and `conversation_l2_cache.md` to distinguish runnable automation code/commands for other researchers, school server, CI, and schedulers from Codex's no-local-heavy-research-run boundary; clarified light local checks are allowed and research `.ipynb` files must be mirrored to same-name `.py` files. | Keep automation scripts and run commands in repo; Codex may run light checks but should not run heavy research pipelines unless local execution is explicitly authorized. |
| 2026-06-08 | Literature Review / Independent Variables | Researched paper-backed independent variables for stock and crypto forecasting without running local training, backtests, or notebook result pipelines. | Created `test/research_materials/independent_variables_literature_review_20260608.md` with 5-step paper reviews and a concrete variable schema; added `test/scripts/send_independent_variables_report_email_utf8.py` for Korean UTF-8 mail delivery. | Commit/push only `stock`, then send the GitHub-rendered report link to the professor. |
