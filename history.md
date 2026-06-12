# 연구 수행 이력 (Research History / State Persistence)

이 파일은 Codex CLI(`Codex CLI`) 에이전트가 새로운 세션으로 넘어가거나 세션이 주기적으로 리셋될 때 **'기억 상실'을 방지하기 위한 영구 저장소(Memory)**입니다.
에이전트는 단일 작업이나 단계를 마무리하기 전, 반드시 `AGENTS.md`에 의거하여 이 파일의 테이블과 요약 섹션을 최신화해야 합니다.

## 📝 History Log

| 날짜 | 단계 | 수행 내용 | 산출물 | 다음 목표 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-05-16 | Phase 0 | 프로젝트 기획안 분석 및 다변량 분석 구조 설계, 아키텍처 다이어그램 업데이트 | `README.md`, `process.md`, `skills.md`, `history.md` 재구성 완료 | Phase 1 (Data Mart 구축) 진입 대기 |
| 2026-05-18 | Phase 2 / Test | 시계열 예측 모델 성능 비교 분석 수행 및 보고서 작성 | `test/1_time_series_test.ipynb`, `test/analysis_report.md`, `README.md` 업데이트 | Phase 2 (다변량 분석 엔진) 고도화 및 감성 데이터 결합 |
| 2026-05-19 | Phase 2 / Advance | 고도화된 단변량 분석 (DuckDB 연동, TCN, XGBoost, 가중 앙상블) | `test/2_time_series_advance_test.ipynb`, `upbit_data.db` | 다변량 분석 및 뉴스 데이터 결합 |
| 2026-05-20 | Phase 2 / Reporting | 유틸리티 확충, MCP 논문 탐색, 모델 해석 고도화 및 리포트 자동화 | `test/scripts/*.py`, `test/results/advanced_analysis_report.md` | 다변량 분석(Phase 2.3) 및 감성 데이터 결합 |
| 2026-05-20 | Env Migration | Codex migration에서 Codex CLI(`Codex CLI`)로 이식 및 규칙 전사| 2026-05-25 | Phase 2 / Benchmark Run | SOTA 15종 시계열 딥러닝 모델의 500분(8.3시간) 전수 학습 및 검증 완료. Autoformer(RMSE 32만, 1위) 및 PatchTST(RMSE 40만, 3위)의 압도적 우위 입증, 통계적 잔차 진단 및 한글 종합 학술 리포트 생성 완료 | `test/results/advanced_analysis_report.md`, `test/images/2_time_series_advance_test_plot_*.png` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-25 | Phase 2 / Multi-Ticker | 업비트 전체 250여 개 종목 대상 동적 변동성/거래대금 필터 결합, 1년 학습/3개월 테스트 단일 Hold-out 기반 SOTA 15종 아키텍처 초고속 딥러닝 학습 및 예측 파이프라인(MinMaxScaler 단일 채택) 구축 완료 | `test/models/3_time_series_multi_ticker_test.ipynb`, `test/models/3_time_series_multi_ticker_test.py`, `test/experiment_specs/3_time_series_multi_ticker_test_framework_design_20260525_214347.md`, `test/images/3_multi_ticker_forecast.png` | Phase 3 (교수님 퀀트 시뮬레이션 및 다변량 DTW 엔진 융합) |
| 2026-05-28 | Phase 3 / Simulation | Upbit 3년 15분봉 데이터마트(DuckDB) 연동 및 수수료 0.05% + 슬리피지 0.02%가 완벽 반영된 실시간 모의 투자 시뮬레이터 v2 집행, Codex AI의 실시간 캔들 패턴(도지, 장대양봉, 상승장악형, 유성선) 직접 판독 의사결정 기록, 이메일 보고 자동 전달 완료 | `pipelines/simulate_and_send.py`, `analysis_report.md`, `test/scripts/send_email.py` | Phase 3 / Step 3.3 (실전 백테스팅 파라미터 튜닝) |
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
  - **에이전트 주의사항:** 시뮬레이션 스크립트는 `pipelines/simulate_and_send.py`에 선언되어 있어 언제든 다시 기동이 가능하며, `.env`는 안전하게 세팅되어 있어 이메일 발송에 아무런 장애가 없습니다.
 
---
*(AI 에이전트에게: 작업을 종료할 때 이 하단 요약 내용과 위 테이블을 수정하여 다음 에이전트가 완벽히 이어받을 수 있게 하십시오.)*
## 2026-06-07 세션 정리 기록

| 날짜 | 단계 | 수행 내용 | 산출물 | 다음 목표 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-07 | MCP 복구 | 공식 GitHub `arxiv-mcp-server`를 `uv tool install --managed-python --python 3.12 git+https://github.com/blazickjp/arxiv-mcp-server.git`로 설치하고, `.agents/mcp_config.json`, `.codex/config.toml`, `.mcp.json`의 실행 경로를 직접 `arxiv-mcp-server` 실행 파일로 표준화했다. | 실제 MCP 서버 바이너리가 PATH에 올라갔고 stdio 프로세스로 정상 기동된다. | Codex MCP loader가 `arxiv-mcp-server`를 인식하는지 확인한 뒤 논문 검색을 재개한다. |
| 2026-06-07 | 마이그레이션 점검 | 오래된 test 문서를 현재 Codex 워크플로우에 맞게 다시 작성하고, `test/` 아래 명백한 임시 생성물을 제거했다. | test 문서가 현재 Codex 워크플로우와 일치하고, `.env`, `__pycache__`, `temp_summary.txt` 같은 불필요 파일이 제거됐다. | 다음 연구 pass 중 임시 출력물이 다시 생기면 계속 정리한다. |
| 2026-06-07 | 임시 Upbit 메일 준비 | `quantitative_trading/AI_trading_temporary/analysis_report.md`에 1,000,000 KRW 임시 Upbit 현물 백테스트 보고서를 생성하고, 커밋 `257ff12`로 `origin/stock`에 푸시해 GitHub 렌더링 링크를 만들었다. | 보고서 링크가 생성됐다: `https://github.com/tabjun/personal_ai_project/blob/stock/quantitative_trading/AI_trading_temporary/analysis_report.md`. 백테스트 결과는 최종 잔고 `2,954,964 KRW`, MDD `-70.45%`였다. | `NAVER_EMAIL_ID`, `NAVER_APP_PASSWORD`, `RECEIVER_EMAIL` 자격 정보가 있으면 `send_email.py`로 메일을 발송한다. |
| 2026-06-07 | 메일 발송 | `test/.env`를 로드하고 `test/scripts/send_email.py`에서 Naver 발신자 ID를 유효한 이메일 주소로 정규화한 뒤 보고서를 발송하고, `test/` 아래 임시 첨부 사본을 제거했다. | Naver SMTP 메일 발송이 성공했고 `test/` 아래 임시 보고서 파일은 남지 않았다. | 새 메일 발송이 필요하지 않으면 임시 보고서는 `AI_trading_temporary/`에만 유지한다. |
| 2026-06-07 | 메일 재발송 수정 | `test/scripts/send_email.py`의 깨진 메일 본문을 UTF-8 한국어 plain-text 메시지로 교체하고, 오래된 임시 보고서를 제거한 뒤 `AI_trading_temporary/AI_trading_youtube_upbit_report_20260607.md`를 생성해 커밋/푸시 `9f75ff8` 후 첨부 없이 GitHub 렌더링 링크로 재발송했다. | 메일 재발송이 성공했고 현재 보고서 링크는 `https://github.com/tabjun/personal_ai_project/blob/stock/quantitative_trading/AI_trading_temporary/AI_trading_youtube_upbit_report_20260607.md`다. | 링크 기반 전달 흐름을 유지한다. |
| 2026-06-08 | 임시 코드 정리 | `test/scripts/send_email.py`를 이전 tracked 내용으로 복원하고 cleanup commit `99e8850`을 푸시해 임시 메일/코드 변경을 history에서 정리했다. | `AI_trading_temporary/`에는 커밋된 보고서 `AI_trading_youtube_upbit_report_20260607.md`만 남았고, ad-hoc AI trading 분석 코드는 남지 않았다. | 새 시뮬레이션 요청이 없으면 임시 산출물은 보고서만 유지한다. |
| 2026-06-08 | Phase 2.2 / 텍스트 컨텍스트 | 뉴스, 시장 리포트, SNS형 로컬 export를 위한 실시간 텍스트 컨텍스트 ingestion 및 feature pipeline을 구축했다. RSS/Naver/local CSV 수집, lexicon 감성, 토픽 flag, 15분 DuckDB factor table, simulation-side risk guard를 추가했다. | `contexts/text_context.py`, `pipelines/ingest_text_context.py`를 추가하고 `pipelines/simulate_and_send.py`, `README.md`를 갱신했다. DuckDB에 `text_events_raw`, `text_features_15m`를 만들었고 RSS 90건 refresh, raw text 91행, feature 500행 생성을 확인했다. | 실시간 가격 캔들을 refresh해 새 텍스트 timestamp와 시장 데이터가 겹치게 한 뒤, text factor를 반영한 no-email 백테스트/보고서 pass를 실행한다. |
| 2026-06-08 | 세션 메모리 / L2 캐시 | 전체 채팅 이력을 읽지 않아도 사용자 의도를 복원할 수 있도록 압축 요청 로그를 추가했다. | `conversation_l2_cache.md`를 만들고 압축 요청/조치/다음 단계 행과 유지 규칙을 추가했다. | 의미 있는 작업 또는 방향 전환 후 압축 행을 하나씩 추가한다. |
| 2026-06-08 | 실행 주체 분리 명확화 | repo-level 자동화는 보존하면서 Codex 로컬 실행 제한은 무거운 연구 실행에만 적용되도록 프로젝트 실행 규칙을 수정했다. | `AGENTS.md`, `skills.md`, `conversation_l2_cache.md`에 다른 연구원/학교 서버/CI/scheduler용 실행 명령과 Codex의 로컬 heavy research run 금지 경계를 분리해 반영했다. 가벼운 로컬 check는 허용하고 연구 `.ipynb`는 동명 `.py` 미러가 필요하다고 명확히 했다. | 자동화 스크립트와 실행 명령은 저장소에 보존하되, Codex는 사용자가 승인하지 않는 한 무거운 연구 pipeline을 실행하지 않는다. |
| 2026-06-08 | 문헌조사 / 독립변수 | 로컬 학습, 백테스트, 노트북 결과 pipeline을 실행하지 않고 주식/코인 예측용 논문 기반 독립변수를 조사했다. | `test/research_materials/independent_variables_literature_review_20260608.md`를 작성했다. 논문별 5단계 리뷰와 구체적 변수 스키마를 포함했고, 커밋/푸시 `216175b` 후 GitHub 렌더링 링크를 교수님께 UTF-8 한국어 메일로 발송했다. | DuckDB feature mart를 확장할 때 해당 변수 스키마를 구현 대상으로 사용한다. |
| 2026-06-08 | Historical Flow 데이터마트 | 과거 유사 사건을 차트 형태만이 아니라 가격 형태, 독립변수 상태, 컨텍스트/인과 요인으로 매칭하는 KRW-wide historical analog mart를 설계하고 구현했다. | `marts/historical_flow.py`, `pipelines/build_historical_flow_mart.py`, `pipelines/query_historical_flows.py`, `test/research_materials/historical_flow_datamart_research_20260608.md`, UTF-8 메일 스크립트를 추가했다. `py_compile`과 tiny synthetic KRW multi-ticker build/query를 통과했고, 커밋/푸시 `699e65e` 후 교수님께 메일 발송했다. full-market build는 로컬에서 실행하지 않았다. | 학교 서버 또는 자동화 환경에서 full mart build를 실행한 뒤 query 결과를 보고서/백테스트에 연결한다. |
| 2026-06-08 | 프레임워크 구조 재정리 | root-level framework module을 역할 기반 package로 재구성해 실제 runtime code와 `test/` 연구 자산을 분리하고 프로젝트 root를 정리했다. | 모듈을 `analysis/`, `advisors/`, `integrations/`, `contexts/`, `marts/`, `pipelines/`로 이동했다. import, command path, README/AGENTS/skills, 메일/보고서 reference를 갱신했고 `py_compile` 및 direct import check를 통과했다. | 독립변수 또는 historical analog 기능을 확장할 때 `/test` 연구 스크립트와 노트북이 `contexts.*`, `marts.*`를 import하도록 정리한다. |
| 2026-06-09 | 도구 설정 / Meta Harness 설치 | Codex에서 사용할 수 있는 Meta Harness 스킬을 이번 저장소 전용으로 설치했다. | `SaehwanPark/meta-harness`를 임시 디렉터리에 clone한 뒤 `--scope project --layout codex`로 설치했다. `.agents/skills/harness`와 `.codex/skills/harness`가 생성되었고, 설치된 `SKILL.md` 메타데이터를 확인했다. | Codex를 재시작하거나 새 세션을 연 뒤, 반복 업무별 스킬/팀 구조가 필요할 때만 `$harness`를 호출한다. |
| 2026-06-09 | 도구 설정 / Meta Harness 전역 설치 | 모든 프로젝트에서 `$harness`를 사용할 수 있도록 Meta Harness를 user scope로도 설치했다. | 기존 임시 clone의 installer를 사용해 `--scope user --layout codex`로 설치했다. `C:\Users\jun99\.agents\skills\harness`와 `C:\Users\jun99\.codex\skills\harness`가 생성되었고 `SKILL.md` 메타데이터를 확인했다. | Codex를 재시작하거나 새 세션을 열면 전역 Harness 스킬을 사용할 수 있다. |
## 2026-06-09 텍스트 독립변수 모델 범위 수정 기록

| 날짜 | 단계 | 수행 내용 | 산출물 및 결과 | 다음 목표 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-09 | Phase 2.2 / 대표 텍스트 모델 | 4번 연구 스크립트가 기존 15개 알고리즘 전체를 재사용하지 않도록 범위를 수정했다. | `test/models/4_text_independent_variable_analysis.py`를 대표 계열 소형 점검용으로 재작성했다. 모델은 `LSTMRepresentative`, `TransformerRepresentative`, `MambaLite`, `TimeXerLite`, `ITransformerLite` 5개이며 기본값은 의도적으로 작게 설정했다(`max_rows=1200`, `seq_len=32`, `epochs=1`, train/test window 제한). `pyproject.toml`에 `torch`를 추가했고 `py_compile`을 통과했다. 무거운 로컬 모델 실행은 하지 않았다. | 승인된 서버 환경에서 빠른 소형 점검을 먼저 실행하고, DA/MASE 정상 범위를 확인한 뒤 row/window/epoch를 키운다. |
| 2026-06-09 | Phase 2.2 / 정상성 및 OOM 안전장치 | 4번 텍스트 독립변수 실험이 비정상 가격 레벨 shortcut 학습과 메모리 실패에 덜 취약하도록 보강했다. | 기본 모델 입력에서 raw `close`를 제거하고 log-return target과 KRW 복원 평가 구조를 유지했다. CUDA OOM 시 batch size를 줄이는 적응형 재시도, batch inference, gradient clipping, 선택적 `--optuna-trials` 소규모 튜닝과 pruning/gc를 추가했다. `pyproject.toml`에 `optuna`를 추가했고 `py_compile`을 통과했다. 무거운 로컬 학습/백테스트는 하지 않았다. | 승인된 실행 환경에서 소형 기본값으로 시작하고, 이후에만 rows/windows/epochs 또는 Optuna trials를 늘린다. |

## 2026-06-09 텍스트 독립변수 분석 모델 추가 기록

| 날짜 | 단계 | 수행 내용 | 산출물 및 결과 | 다음 목표 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-09 | Phase 2.2 / 텍스트 독립변수 분석 모델 | 실시간 뉴스, 증시 리포트, SNS 텍스트 변수를 독립변수로 사용해 가격 예측을 분석하는 다음 연구 스크립트를 `test/models` 아래 추가했다. | `test/models/4_text_independent_variable_analysis.py`를 생성했다. DuckDB 캔들과 `text_features_15m`을 결합하고, OHLCV/변동성/기술지표/유동성 피처와 텍스트 컨텍스트 피처를 구성한다. 가격 기준선과 텍스트 반영 ridge 예측을 비교하며 KRW 스케일 RMSE/MAE, DA, MASE, 단순 persistence MAE를 보고한다. `py_compile`을 통과했고 무거운 로컬 실행은 하지 않았다. | 학교 서버 또는 승인된 실행 환경에서 스크립트를 실행한 뒤, 생성되는 `test/results/4_text_independent_variable_*` 리포트로 텍스트 컨텍스트가 기준선을 개선하는지 판단한다. |
## 2026-06-09 노트북 미러 Hook 기록

| 날짜 | 단계 | 수행 내용 | 산출물 및 결과 | 다음 목표 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-09 | 저장소 워크플로우 / 노트북 미러 Hook | 연구용 `.ipynb` 변경 시 GitHub diff 확인이 가능한 동명 `.py` 미러를 반드시 유지하도록 규칙을 자동화했다. | `test/scripts/sync_ipynb_mirrors.py`와 `.githooks/pre-commit`을 추가하고, 로컬 Git에 `core.hooksPath=quantitative_trading/.githooks`를 설정했다. Hook은 stage된 노트북을 실행하지 않고 파싱해 같은 경로의 `.py` 미러를 작성하고 자동 stage한다. `py_compile`을 통과했다. | 노트북 변경 커밋 시 자동 동기화된 `.py` diff를 노트북과 함께 리뷰한다. |
| 2026-06-09 | 저장소 워크플로우 / 노트북 미러 위치 수정 | Hook 위치를 바로잡고 4번 실험을 노트북과 동명 `.py` 미러 구조로 복원했다. | Hook helper를 `test/scripts`에서 `.githooks/sync_ipynb_mirrors.py`로 옮기고 `.githooks/pre-commit`을 갱신했다. `test/models/4_text_independent_variable_analysis.ipynb`를 생성하고 hook helper로 동명 `.py` 미러를 재생성했으며, 적응형 정상성 진단을 위해 `statsmodels`를 추가했다. `py_compile`을 통과했다. | 워크플로우 유틸리티는 `test/scripts` 밖에 두고, 향후 연구 실험은 `.ipynb`와 동명 `.py` 미러를 함께 유지한다. |
## 2026-06-09 커밋 이력 기반 워크플로우 보호 규칙 기록

| 날짜 | 단계 | 수행 내용 | 산출물 및 결과 | 다음 목표 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-09 | 저장소 워크플로우 / 보호 규칙 | initial commit부터 2026-05-28까지의 워크플로우 규칙을 복원하고, 임시 `test/scripts` 난립을 막는 자동 보호 장치를 추가했다. | `AGENTS.md`와 `test/README.md`를 갱신했다. `.githooks/check_repo_policy.py`를 추가하고 `.githooks/pre-commit`이 노트북 미러 동기화와 저장소 정책 검사를 함께 실행하도록 수정했다. 생성된 `__pycache__`를 제거했으며, 정책은 새 `test/scripts/*.py` 추가와 동명 `.ipynb` 없는 `test/models/*.py`를 차단한다. 검증을 통과했다. | 연구 추가 작업은 노트북을 원본으로 두고 동명 `.py` 미러를 유지하며, 일회성 스크립트 생성 대신 기존 유틸리티를 재사용/확장한다. |
| 2026-06-09 | 교수님 설명용 데이터마트 메일/첨부 제작 | 모델 내부 수정이 아닌 분석용 외생변수 선설계라는 점을 한국어로 명확히 설명하고, PDF는 먼저 로컬 렌더링으로 검증한 뒤 메일 첨부로 전송해야 함. | `test/scripts/send_historical_flow_datamart_professor_email_utf8.py`를 추가했다. 과거 흐름 기반 데이터마트의 정의, 후보군(구간별/일정별/혼합형), 예시 시각화 3종을 담은 한국어 HTML/PDF를 생성했고, `test/results/historical_flow_datamart_professor_brief_20260609.pdf`를 먼저 확인한 뒤 Naver SMTP로 교수님께 발송했다. | 교수님 피드백을 받으면 구간 분할 기준과 혼합형 마트 스키마를 그에 맞춰 좁힌다. |
| 2026-06-09 | 저장소 정리 / 임시 메일 산출물 제거 | 메일 발송용 임시 파일과 repo-local harness 참조 문서를 제거하고, 작업 지침 문서를 루트 기준으로 다시 모았다. | `test/skills.md`, `test/scripts/send_historical_flow_datamart_professor_email_utf8.py`, `test/results/historical_flow_datamart_professor_brief_20260609.{html,pdf,png}`를 제거했다. repo-local `.agents/skills/harness`, `.codex/skills/harness`도 삭제해 중복 참조를 정리했고 Git stage는 비웠다. | 다음 세션에서는 루트 `AGENTS.md`, `skills.md`, `history.md`, `process.md`만 우선 읽고 `/test` 분석 작업을 이어간다. |
| 2026-06-09 | 저장소 문서 구조 재정리 | 영구 규칙과 최근 요청 캐시가 섞여 길어지던 문서 구조를 역할별로 다시 나눴다. | `AGENTS.md`를 강제 규칙 중심으로 재작성하고, `skills.md`는 기술 철학과 분석 기준 중심으로 재구성했다. `README.md`는 사람용 프로젝트/실행 가이드로 정리했고, `conversation_l2_cache.md`는 최근 요청 5개만 남기는 얇은 캐시로 축소했다. | 다음 세션부터는 루트 문서 4종과 짧은 캐시만 읽고 빠르게 복원한다. |
| 2026-06-09 | 테스트 스크립트 정리 및 캐시 확장 | `test/scripts`를 실사용 기준으로 줄이고, L2 캐시 보존 개수를 세션 체감에 맞춰 확대했다. | `conversation_l2_cache.md`의 유지 원칙을 최근 요청 최대 20개로 수정했다. `test/scripts/send_email.py`를 preset 기반 공용 UTF-8 메일 모듈로 통합했고, `extract_notebook_images.py`만 노트북 후처리 유틸로 남겼다. `build_*`, `enhanced_report_generator.py`, `notebook_to_md.py`, `reconstruct_test_env.py`, 개별 UTF-8 메일 스크립트들을 제거했다. `test/data/.gitkeep`를 추가하고 `test/README.md`에 데이터 폴더와 남는 스크립트 역할을 문서화했다. | 이후 `/test` 보조 유틸은 공용 모듈만 유지하고, 새 요청이 와도 ad-hoc 스크립트 대신 기존 모듈 확장으로 처리한다. |
| 2026-06-13 | 5번 최적화 진단 정리 | `test/models/5_optimization_diagnostics_test.py`를 5개 대표 아키텍처 비교 구조로 유지하고, `test/README.md`는 진짜 대문 수준으로 축약했다. | `quick_probe`, `architecture_probe`, `full_matrix`가 `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`를 실제로 비교하도록 맞췄고, `test/README.md`의 상세 설명은 결과 보고서와 노트북으로 이동했다. | 이후 세션에서는 실험 설명은 `test/results/*.md`, 실행 구조는 `test/models/*.ipynb`와 `test/models/*.py`, 요약은 `test/README.md`를 우선 본다. |
| 2026-06-13 | research-workflow harness | 반복되는 연구/문서/보고서 흐름을 다음 세션에서 바로 복원할 수 있도록 최소 harness 구조를 추가했다. | `docs/harness/research-workflow/team-spec.md`를 추가했고, `.agents/skills/research-workflow-orchestrator/SKILL.md`를 생성했다. `AGENTS.md`에는 이 team spec을 우선 읽도록 짧은 포인터를 넣었다. | 다음 세션에서는 반복 작업 시 `AGENTS.md -> team-spec -> orchestrator skill` 순서로 복원한다. |

## 2026-06-13 5번 최적화 진단 복구 기록

- [x] `test/models/5_optimization_diagnostics_test.py`의 `architecture_probe`와 `full_matrix`를 대표 아키텍처 5개(`Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`) 기준으로 복구했다.
- [x] 같은 변경을 `test/models/5_optimization_diagnostics_test.ipynb` 코드 셀에도 반영해 노트북 원본과 `.py` 미러의 의미를 다시 맞췄다.
- [x] 보고서 템플릿의 아키텍처 해설도 `TCN`/`Transformer` 비교군을 포함하도록 확장했다.
- [ ] 다음 체크포인트: 승인된 실행 환경에서만 필요 시 노트북을 다시 실행해 출력 셀까지 최신화한다.

## 2026-06-13 5번 진단 범위 전체 복구

- [x] `quick_probe`와 `objective_probe`까지 `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer` 5개 대표군으로 확장했다.
- [x] 보고서 초록에서 `LSTM 기준` 서술을 제거하고 5개 대표군 비교로 통일했다.
- [x] 노트북 출력은 전부 비워서 재실행 전 상태를 깔끔하게 정리했다.
- [ ] 다음 체크포인트: 승인된 환경에서만 필요 시 실제 결과를 다시 생성한다.
| 2026-06-13 | quick_probe 보고서 정리 | 저장본이 LSTM quick_probe 요약이라는 점과 최신 5개 대표군 결과는 별도 재생성 대상이라는 점을 분리해야 했다. | `test/results/5_optimization_diagnostics_quick_probe_20260613.md` 초록과 결론을 현재 저장본 기준으로 정리했다. | 최신 5개 대표군 결과가 다시 저장되면 보고서를 그 수치로 덮어쓴다. |
| 2026-06-13 | quick_probe 보고서 서식 전환 | 현재 `ipynb` 출력이 비어 있어 결과 수치 대신 5개 대표군 기준 서식이 필요했다. | `test/results/5_optimization_diagnostics_quick_probe_20260613.md`를 `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer` 전체 비교용 보고서 틀로 재작성했다. | 다음 실행 결과가 생기면 같은 서식에 실제 수치를 채운다. |
| 2026-06-13 | quick_probe 보고서 로컬 출력 복구 | 보고서 source of truth는 현재 로컬 노트북 출력이어야 하므로, 임시 서식 대신 실제 출력 Markdown과 이미지 번들을 다시 반영해야 했다. | 현재 `test/models/5_optimization_diagnostics_test.ipynb`의 display markdown을 `test/results/5_optimization_diagnostics_quick_probe_20260613.md`로 다시 반영했고, `test/scripts/extract_notebook_images.py`로 PNG 34장을 `test/images/`에 추출했다. | 이후 5번 보고서 수정은 항상 현재 로컬 `.ipynb` 출력과 추출 이미지 기준으로만 이어간다. |
| 2026-06-13 | 연구 기록 추적성 원칙 명확화 | 최근 요청 캐시는 얇게 유지하되, 연구 시작부터의 문제 제기·방법론 수정·보고서 해석 원칙 변화는 커밋 이력과 이력 문서만 봐도 복원 가능해야 한다는 방향을 명확히 했다. | `conversation_l2_cache.md`에는 최근 의도만 압축 유지하고, 중요한 워크플로우/해석 원칙 변화는 `history.md`와 의미 있는 커밋 메시지에도 남기는 구조를 확정했다. | 이후 세션에서는 캐시를 비대하게 늘리지 않고, 대신 커밋 단위와 history log만으로도 논문용 연구 흐름을 되짚을 수 있게 기록 품질을 유지한다. |
| 2026-06-13 | 5번 최종 보고서 및 6번 후속 안정화 계획 | 5번 quick_probe 보고서를 교수님 전달용 최종 문서로 다듬고, 15개 케이스 상세 해석을 `무엇을 시험했나 / 좋은 그림 기준 / 이번 그림 해석 / 결과 해석` 구조로 통일했다. 최적화 문제가 해결되어야 독립변수·데이터마트 확장이 해석 가능하다는 다음 단계 원칙도 정리했다. | `test/results/5_optimization_diagnostics_quick_probe_20260613.md`를 최종 보고서로 보강했고, `test/experiment_specs/6_optimization_stabilization_plan_20260613.md`, `test/models/6_optimization_stabilization_test.ipynb`, `test/models/6_optimization_stabilization_test.py`를 추가했다. `test/scripts/send_email.py`의 `optimization_context_brief` preset도 이번 보고서/계획 링크 중심으로 갱신했다. | 커밋/푸시 후 `develop`, `main`에 병합하고, UTF-8 메일 preset으로 교수님께 최종 보고 링크를 발송한다. |
| 2026-06-13 | 6번 최적화 안정화 실험 구체화 | 5번은 진단 기준선이고 6번은 해결 실험이라는 역할을 분리했다. 5번에는 normalization ablation과 stabilization loss suite를 추가했고, 6번은 OOM 재시도와 stage별 실행 계획을 가진 서버용 오케스트레이터로 확장했다. | `test/models/5_optimization_diagnostics_test.ipynb/.py`에 `--normalization`, `--num-workers`, `stabilization_loss_probe`를 추가했다. `test/models/6_optimization_stabilization_test.ipynb/.py`는 `server_1024`, `server_2048`, `server_light` 프로필, `--run-stage`, `--run-all`, `--dry-run`, CUDA OOM batch-size retry를 지원한다. `test/experiment_specs/6_optimization_stabilization_plan_20260613.md`에는 5번 수정 이유, 문헌 근거, stage별 좋은/나쁜 그림 기준, 실행 명령을 보강했다. | 승인된 서버 환경에서 `uv run test/models/6_optimization_stabilization_test.py --run-all --profile server_2048`로 실행하고, 생성된 CSV/이미지/노트북 출력 기준으로 6번 보고서를 작성한다. |
| 2026-06-13 | 예측 방법론 문헌 리뷰 및 교수님 공유 준비 | 3번 Autoformer 착시, 4/5/6번 연구 흐름, 최근 시계열 예측 알고리즘, 금융·코인 논문에서 최적화/평가 문제가 들어가는 위치를 문헌 기반으로 정리했다. | `test/research_materials/forecasting_methodology_literature_review_20260613.md`를 추가했고, `test/README.md`에 4/5/6번 역할과 문헌 리뷰 포인터를 연결했다. `test/scripts/send_email.py`에는 `forecasting_methodology_review` UTF-8 메일 프리셋을 추가했다. | 커밋/푸시 후 교수님께 문헌 리뷰 렌더링 링크를 발송하고, 다음 단계는 6번 서버 실행 결과 기준 보고서 작성이다. |
