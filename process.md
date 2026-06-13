# 연구 수행 프로세스 (Research Process)

본 문서는 긴 호흡의 AI 에이전트 세션(Codex CLI 등)이 토큰 한계로 인해 망가지는 것을 방지하고, 사용자의 주기적 세션 초기화 요구에 완벽하게 대응하기 위해 작업을 모듈화된 **'단위 프로세스(Step)'**로 나눈 가이드라인입니다.

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
## MCP 복구 기록

- [x] `.agents/mcp_config.json`, `.codex/config.toml`, `.mcp.json`의 arXiv MCP 실행 대상을 `arxiv-mcp-server`로 맞췄다.
- [x] `README.md`와 `AGENTS.md`의 문서화된 실행 경로도 설치된 서버 바이너리와 일치하도록 갱신했다.
- [ ] Codex MCP loader가 `arxiv-mcp-server`를 인식하는지 확인하는 것을 다음 체크포인트로 유지한다.

## 2026-06-07 세션 정리 기록

- [x] 현재 Codex / `arxiv-mcp-server` 워크플로우에 맞게 테스트 문서를 정리했다.
- [x] `test/` 아래의 명백한 임시 생성물을 정리하고 유용한 실험 산출물만 남겼다.
- [ ] 다음 연구 작업 중 임시 출력물이 다시 생기면 파일 트리 점검을 다시 수행한다.

## 2026-06-07 임시 메일 준비 기록

- [x] 임시 Upbit 현물 백테스트 보고서를 `quantitative_trading/AI_trading_temporary/analysis_report.md`에 생성했다.
- [x] 이메일 전달용 GitHub 링크를 만들기 위해 보고서를 `origin/stock`에 커밋/푸시했다.
- [x] `test/.env`를 로드한 뒤 `test/scripts/send_email.py`로 메일을 발송했다.
- [x] 발송 후 `test/` 아래 임시 첨부 사본을 제거했다.
- [x] 깨진 메일 템플릿을 UTF-8 한국어 링크 전용 본문으로 교체했다.
- [x] 오래된 임시 보고서를 제거하고 `AI_trading_youtube_upbit_report_20260607.md` 링크로 수정 메일을 재발송했다.
- [x] 재발송 후 `test/scripts/send_email.py`를 복원해 임시 AI trading 산출물은 커밋된 Markdown 보고서만 남겼다.

## 2026-06-08 실시간 텍스트 컨텍스트 기록

- [x] RSS, 선택적 Naver News API, 로컬 리포트/SNS CSV export에서 실시간 텍스트를 수집하는 `contexts/text_context.py`를 추가했다.
- [x] 15분 독립변수 테이블 `text_events_raw`, `text_features_15m`를 DuckDB mart에 추가했다. 이벤트 수, 감성 평균/합계, shock z-score, 1시간 감성 모멘텀, 매크로/리스크/크립토/규제/유동성 토픽 count를 포함한다.
- [x] 운영 엔트리포인트 `pipelines/ingest_text_context.py`를 추가했다: `uv run pipelines/ingest_text_context.py`.
- [x] `pipelines/simulate_and_send.py`에 텍스트 factor를 연결하고, 부정적/리스크성 텍스트 컨텍스트가 신규 진입을 차단할 수 있도록 `text_risk_guard`를 추가했다.
- [x] ingestion end-to-end 확인: RSS 90건 refresh, raw text 91행 유지, candle-aligned feature 500행 생성.
- [ ] 다음 체크포인트: 실시간 Upbit candle mart를 refresh/확장해 2026-06-08 텍스트 timestamp와 가격 bucket이 겹치게 한 뒤, 이메일 발송 없이 text-aware 백테스트/보고서 pass를 실행한다.

## 2026-06-08 대화 L2 캐시 기록

- [x] 프로젝트 루트에 `conversation_l2_cache.md`를 추가했다.
- [x] 압축 로그 형식 `날짜`, `요청`, `제약`, `조치/산출물`, `다음 단계`를 정의했다.
- [x] 실시간 텍스트 컨텍스트 요청과 L2 캐시 요청을 초기 기록으로 넣었다.
- [ ] 의미 있는 사용자 작업이나 방향 전환마다 압축 행을 하나씩 추가한다. 전체 원문 대화는 저장하지 않는다.

## 2026-06-08 실행 주체 분리 기록

- [x] 다른 연구원/운영자를 위해 repo-level 자동화 코드, `uv run ...` 예시, n8n/Cron/CI/Docker/Kubernetes 설계, 학교 서버 실행 절차는 저장소에 남겨야 함을 명확히 했다.
- [x] 제한 대상은 Codex가 사용자 개인 연구 세션에서 로컬 터미널 또는 `.venv`로 장시간 분석, 학습, 백테스트, 노트북 결과 산출 파이프라인을 실행하는 경우임을 명확히 했다.
- [x] 단순 계산, 컴파일/문법 확인, import 확인, 작은 단위 테스트 같은 가벼운 로컬 점검은 허용된다고 명확히 했다.
- [x] 연구용 `.ipynb` 생성/수정 시 추적성을 위해 동명 `.py` 미러를 함께 유지해야 함을 명확히 했다.
- [x] `AGENTS.md`, `skills.md`, `conversation_l2_cache.md`에 분리 규칙을 반영했다.
- [ ] 향후 자동화를 추가할 때 실행 가능한 명령은 문서화하되, 사용자가 명시적으로 승인하지 않는 한 Codex는 가벼운 로컬 점검만 수행한다.

## 2026-06-08 독립변수 문헌조사 기록

- [x] 조사 범위를 주식/코인 예측 독립변수 설계로 한정하고, 로컬 대용량 학습/백테스트/노트북 실행은 수행하지 않았다.
- [x] arXiv 기반 논문을 참고하여 가격/변동성, 기술지표, 유동성, 오더북, 뉴스/리포트/SNS 감성, 한국어 금융 텍스트, 매크로/레짐, 글로벌 유동성, 온체인/파생상품, 크로스마켓, 이벤트 쇼크 독립변수 후보를 정리했다.
- [x] `test/research_materials/independent_variables_literature_review_20260608.md`에 논문별 5단계 포맷(요약, 서론, 분석 기법, 결과, 결론 및 설계 결정)을 적용했다.
- [x] 한글 깨짐 방지를 위해 `test/scripts/send_independent_variables_report_email_utf8.py`를 UTF-8/base64 MIME 본문 방식으로 작성했다.
- [x] `stock` 브랜치에만 커밋/푸시한 뒤 교수님께 GitHub 렌더링 링크를 UTF-8 한글 이메일로 발송했다.

## 2026-06-08 Historical Flow 데이터마트 기록

- [x] 과거 유사 사건/흐름 데이터마트를 `KRW-BTC` 단일 종목이 아니라 업비트 KRW 마켓 전체 `ticker` 축 기준으로 설계했다.
- [x] 전체 KRW 원천 window는 저장하고, 기본 neighbor index는 거래대금 상위 유동성 종목 subset으로 구성하도록 했다.
- [x] 단순 차트 형태 비교를 피하기 위해 `shape_distance`, `factor_distance`, `context_distance`를 합친 복합 유사도 구조를 반영했다.
- [x] 변곡점 당시 원인/상황 파악을 위해 `text_features_15m`의 감성, 이벤트 수, shock z-score, 리스크/매크로/규제/유동성 토픽을 optional context vector로 결합했다.
- [x] 리서치 보고서는 `test/research_materials/historical_flow_datamart_research_20260608.md`에 표준 5단계 포맷으로 작성했다.
- [x] 문법 검사와 tiny synthetic KRW multi-ticker 검증으로 table build/query 및 `query_composite_distance` 생성을 확인했다.
- [x] `stock` 브랜치에만 커밋/푸시한 뒤 교수님께 UTF-8 한글 이메일로 보고했다.

## 2026-06-08 프레임워크 구조 재정리 기록

- [x] 루트 경로에 흩어져 있던 실사용 모듈을 역할군별 패키지로 재배치했다.
- [x] `main.py`는 루트에 유지하고, 나머지 실사용 모듈은 `analysis/`, `advisors/`, `integrations/`, `contexts/`, `marts/`, `pipelines/`로 정리했다.
- [x] `contexts/text_context.py`와 `marts/historical_flow.py`가 `/test` 분석 코드에서 직접 import 가능한 실사용 프레임워크 경로가 되도록 맞췄다.
- [x] `README.md`, `AGENTS.md`, `skills.md`, 메일 스크립트, 연구 문서의 실행 경로와 코드 링크를 새 구조에 맞춰 갱신했다.
- [x] `py_compile` 및 import check로 새 패키지 구조의 경로 정합성을 확인했다.
- [ ] 필요 시 다음 단계에서 `/test/models/*.py`와 노트북 import 예시도 새 패키지 기준으로 추가 정리한다.

## 2026-06-09 Meta Harness 프로젝트 로컬 설치 기록

- [x] `SaehwanPark/meta-harness`를 임시 설치 디렉터리로 clone했다.
- [x] 프로젝트 범위 Codex 레이아웃으로 설치했다: `--scope project --layout codex`.
- [x] 공유 스킬 경로 `.agents/skills/harness`와 Codex 네이티브 미러 `.codex/skills/harness`를 생성했다.
- [x] 두 위치의 `SKILL.md`가 `name: harness` 메타데이터를 노출하는 것을 확인했다.
- [ ] 전체 하네스 재구성은 보류한다. 반복 업무가 명확해질 때만 `$harness`로 개별 스킬/팀 스펙을 작게 추가한다.

## 2026-06-09 Meta Harness 전역 설치 기록

- [x] 프로젝트 로컬 설치와 별도로 user scope 전역 설치를 진행했다.
- [x] `--scope user --layout codex`로 설치하여 `C:\Users\jun99\.agents\skills\harness`를 생성했다.
- [x] Codex 네이티브 전역 미러 `C:\Users\jun99\.codex\skills\harness`를 생성했다.
- [x] 두 전역 위치의 `SKILL.md`가 `name: harness` 메타데이터를 노출하는 것을 확인했다.
- [ ] Codex 재시작 또는 새 세션 이후 모든 프로젝트에서 `$harness`를 사용할 수 있다.
## 2026-06-09 텍스트 독립변수 모델 범위 수정 기록

- [x] `test/models/4_text_independent_variable_analysis.py`를 15개 전체 모델 벤치마크가 아니라 대표 계열별 소형 점검 테스트로 재작성했다.
- [x] 모델 세트는 `LSTMRepresentative`, `TransformerRepresentative`, `MambaLite`, `TimeXerLite`, `ITransformerLite` 5개로 축소했다.
- [x] `TimeXerLite`는 외생변수 교차 어텐션, `ITransformerLite`는 변수 토큰 관점, `MambaLite`는 Mamba/SSM 계열의 게이트형 인과 합성곱 프록시로 구현했다.
- [x] 기본 실험 크기를 `max_rows=1200`, `seq_len=32`, `epochs=1`, `max_train_windows=512`, `max_test_windows=128`로 낮춰 아주 거친 1차 확인용으로 만들었다.
- [x] 4번 스크립트 재현 실행에 필요한 `torch`를 `pyproject.toml` 의존성에 명시했다. 설치/학습 실행은 하지 않았다.
- [x] `python -m py_compile test\models\4_text_independent_variable_analysis.py` 문법 검증은 통과했다.
- [ ] 다음 체크포인트: 승인된 서버 환경에서 빠른 소형 점검 실행을 먼저 수행하고, 모델별 DA/MASE가 정상 범위인지 본 뒤 row/window/epoch를 키운다.

## 2026-06-09 정상성/OOM/Optuna 안전장치 기록

- [x] 4번 스크립트 기본 입력에서 비정상 원시 가격 `close`를 제거하고, `log_return_1`, 이동 수익률, 실현 변동성, RSI/ROC/Bollinger, 유동성 대리 변수 등 정상성에 가까운 파생 피처만 기준선으로 쓰도록 수정했다.
- [x] 예측 목표는 다음 종가 자체가 아니라 다음 로그수익률로 학습하고, 평가 단계에서만 `prev_close * exp(pred_return)`으로 KRW 원본 가격을 복원하도록 유지했다.
- [x] CUDA OOM 발생 시 batch size를 절반으로 줄여 재시도하는 적응형 배치 축소 로직을 추가했다.
- [x] 학습/테스트 예측 모두 batch 단위로 수행하도록 바꿔 평가 중 OOM 위험도 낮췄다.
- [x] `grad_clip_norm` 기본값 1.0을 추가해 폭주 기울기와 초기 급격한 업데이트를 완화했다.
- [x] `--optuna-trials` 옵션을 추가하고, 요청 시 작은 검증 split과 `MedianPruner`, `gc_after_trial=True`로 소규모 튜닝만 수행하도록 했다.
- [x] `optuna`를 `pyproject.toml` 의존성에 명시했다. 설치/튜닝/학습 실행은 하지 않았다.
- [x] `python -m py_compile test\models\4_text_independent_variable_analysis.py` 문법 검증은 통과했다.

## 2026-06-13 예측 방법론 문헌 리뷰 기록

- [x] 3번 실험에서 Autoformer가 좋아 보였던 현상을 가격 레벨 복사/평균수렴 착시 가능성으로 재해석했다.
- [x] 최근 시계열 예측 흐름을 Autoformer, FEDformer, PatchTST, DLinear, TimesNet, iTransformer, TimeXer, ModernTCN, Mamba/MambaTS, Chronos/TimesFM/Moirai, TFT, RevIN까지 확장해 정리했다.
- [x] 금융·코인 예측 논문들이 최적화 문제를 data/preprocessing, experimental setup, model training, loss, evaluation, ablation, baseline에 어떻게 포함하는지 정리했다.
- [x] `test/research_materials/forecasting_methodology_literature_review_20260613.md`를 추가하고, `test/README.md`와 `test/scripts/send_email.py`의 `forecasting_methodology_review` preset에 연결했다.
- [ ] 다음 체크포인트: 커밋/푸시 후 교수님께 렌더링 링크를 발송하고, 6번 서버 실행 결과 기준으로 안정화 보고서를 작성한다.

## 2026-06-13 서버 환경 재구성 기록

- [x] 학교 서버 터미널에서 `py`가 없고 `python3.10`만 보이는 상황을 확인했다.
- [x] `uv venv --python 3.12`, `uv sync`, `ipykernel install` 흐름을 `test/README.md`에 추가했다.
- [x] JupyterLab에서 파일별 실행 후 `Kernel -> Shut Down All Kernels`를 반드시 수행하도록 공통 규칙을 문서화했다.
- [ ] 다음 체크포인트: 서버에서 새 kernel을 등록할 때 `uv`가 실제로 3.12를 내려받을 수 있는지, 아니면 3.10 커널로 고정해야 하는지 확인한다.

## 2026-06-13 서버 환경 3.12 고정 기록

- [x] `.python-version`을 `3.12`로 추가해 `uv`가 기본적으로 3.12를 선택하도록 맞췄다.
- [x] `test/README.md`의 서버용 복붙 명령을 `uv venv --python 3.12`, `UV_CACHE_DIR`, `cu126` torch wheel, `quant312` kernel 등록 순서로 다시 정리했다.
- [ ] 다음 체크포인트: 서버에서 `python -m ipykernel install --user --name quant312 ...` 후 JupyterLab에서 해당 커널이 정상 노출되는지 확인한다.

## 2026-06-13 서버 환경 분기 문서화

- [x] `test/README.md`에 `uv` 기준 경로와 `python3.12`가 실제로 설치된 경우의 일반 `venv` 경로를 분리해 넣었다.
- [x] 일반 `venv` 경로는 `uv export --format requirements-txt`로 의존성을 풀어 설치하도록 적었다.
- [ ] 다음 체크포인트: 서버에서 어떤 경로가 가능한지 확인한 뒤 하나만 골라 반복 사용한다.

## 2026-06-13 서버 환경 bootstrap 스크립트 추가

- [x] `test/scripts/bootstrap_uv_313.sh`, `bootstrap_uv_312.sh`, `bootstrap_venv_313.sh`, `bootstrap_venv_312.sh`를 추가했다.
- [x] `test/README.md`는 긴 설치 설명 대신 스크립트 호출 방법과 서버에서의 git pull / commit / push 순서만 남기도록 줄였다.
- [ ] 다음 체크포인트: 서버에서 상황에 맞는 스크립트 하나만 실행해서 kernel 등록까지 끝나는지 확인한다.

## 2026-06-13 서버 환경 이름 충돌 방지

- [x] bootstrap 공통 로직을 `test/scripts/bootstrap_env_common.sh`로 분리했다.
- [x] 모든 bootstrap 스크립트가 고정 `.venv` 대신 `.venvs/<env_name>`에 새 환경을 만들도록 바꿨다.
- [x] 기본 env 이름은 timestamp 기반으로 자동 생성되고, 기존 env/kernels 목록과 제거 명령도 함께 출력되도록 바꿨다.
- [ ] 다음 체크포인트: 서버에서 실제 실행 시 생성된 env 이름과 kernel 이름이 JupyterLab에서 기대대로 보이는지 확인한다.

## 2026-06-13 `ipykernel` 기본 의존성 반영

- [x] `pyproject.toml` 기본 dependencies에 `ipykernel`을 추가했다.
- [x] bootstrap 스크립트는 이미 `ipykernel` 설치를 별도로 하고 있으므로 추가 수정은 하지 않았다.
- [ ] 다음 체크포인트: 서버에서 기존 env에 `uv sync`를 한 번 더 실행한 뒤 `python -m ipykernel install ...`이 바로 되는지 확인한다.

## 2026-06-13 6번 결과 보고와 7번 후속 확장 분리

- [x] 5번을 진단 기준선, 6번을 안정화 오케스트레이터, 7번을 넓은 폭의 후속 확장 실험으로 역할 분리했다.
- [x] 6번은 실제 대규모 결과 leaderboard가 아니라 stage plan과 의사결정 gate를 보존하는 보고서로 정리했다.
- [x] `test/results/6_optimization_stabilization_stage_report_20260613.md`를 추가해 의도/설계/현재 결과/결과 해석/추가 연구 필요성/다음 스텝을 독립 문서로 작성했다.
- [x] 7번 후속 확장 실험 계획서와 notebook+mirror를 `test/experiment_specs/7_optimization_breadth_expansion_plan_20260613.md`, `test/models/7_optimization_breadth_expansion_test.ipynb`, `test/models/7_optimization_breadth_expansion_test.py`로 추가했다.
- [x] 7번은 `breadth_probe`, `ensemble_probe`, `normalization_cross_check`, `loss_cross_check`, `scale_confirmation` suite를 포함하도록 정의했다.
- [x] `AGENTS.md`, `docs/harness/research-workflow/team-spec.md`, `test/README.md`에 “후속 연구는 새 번호”와 “새 보고서는 항상 독립 문서” 규칙을 명시했다.
- [ ] 다음 체크포인트: 커밋/푸시 후 `test/scripts/send_email.py --preset optimization_stabilization_stage`로 교수님 메일을 발송하고, 서버 실행 결과가 생기면 7번 실제 결과 보고서를 작성한다.

## 2026-06-09 텍스트 독립변수 분석 모델 추가 기록

- [x] `test/models/4_text_independent_variable_analysis.py`를 추가해 뉴스/증시 리포트/SNS 텍스트 피처를 독립변수로 쓰는 연구용 분석 엔트리포인트를 만들었다.
- [x] DuckDB 가격 테이블과 `text_features_15m`을 15분 timestamp 기준으로 조인하고, OHLCV/변동성/기술지표/유동성 피처와 텍스트 감성·토픽·shock 피처를 결합하도록 구성했다.
- [x] 가격 기반 기준선과 텍스트 독립변수 포함 피처 묶음을 같은 폐쇄형 ridge 모델로 비교하고, KRW 원본 스케일의 RMSE/MAE, DA, MASE, 단순 persistence MAE를 산출하도록 했다.
- [x] 실행 결과물은 `test/results/4_text_independent_variable_*` CSV/Markdown 리포트로 저장되도록 했다.
- [x] Codex 로컬 세션에서는 장시간 학습/백테스트를 실행하지 않고 `python -m py_compile test\models\4_text_independent_variable_analysis.py` 경량 문법 검증만 통과했다.
- [ ] 다음 체크포인트: 학교 서버 커널 또는 승인된 실행 환경에서 `uv run test/models/4_text_independent_variable_analysis.py --db upbit_data.db --table btc_15m_advance`로 실제 텍스트 반영 비교 리포트를 생성한다.
## 2026-06-09 노트북 미러 Hook 기록

- [x] `.ipynb` 변경 시 동명 `.py` 미러를 항상 생성/갱신하도록 `test/scripts/sync_ipynb_mirrors.py`를 추가했다.
- [x] 추적 가능한 Git hook 경로 `.githooks/pre-commit`을 추가했다.
- [x] `git config core.hooksPath quantitative_trading/.githooks`를 적용해 현재 저장소에서 pre-commit hook이 동작하도록 설정했다.
- [x] hook은 stage된 `.ipynb`를 감지하면 노트북 JSON의 markdown/code cell을 실행 없이 파싱해 같은 경로의 `.py`로 미러링하고 자동 `git add`한다.
- [x] `python -m py_compile test\scripts\sync_ipynb_mirrors.py` 문법 검증은 통과했다.
- [ ] 다음 체크포인트: `.ipynb`를 수정하는 커밋에서는 hook 출력 `[ipynb-mirror] Synced Python mirrors:`를 확인하고, 생성된 `.py` diff도 함께 리뷰한다.

## 2026-06-09 노트북 미러 위치 수정 기록

- [x] hook 보조 스크립트를 연구 스크립트 폴더인 `test/scripts`에서 제거하고 `.githooks/sync_ipynb_mirrors.py`로 이동했다.
- [x] `.githooks/pre-commit`이 `.githooks/sync_ipynb_mirrors.py`를 호출하도록 수정했다.
- [x] 4번 연구 파일을 `test/models/4_text_independent_variable_analysis.ipynb`와 동명 `.py` 미러 구조로 맞췄다.
- [x] 4번 `.py`는 `.ipynb`에서 hook helper의 `notebook_to_python()` 규칙으로 재생성했다.
- [x] 적응형 정상성 진단을 위해 `statsmodels`를 `pyproject.toml` 의존성에 명시했다.
- [x] `python -m py_compile test\models\4_text_independent_variable_analysis.py .githooks\sync_ipynb_mirrors.py` 검증은 통과했다.
## 2026-06-09 커밋 이력 기반 워크플로우 보호 규칙 기록

- [x] initial commit부터 2026-05-28까지의 커밋 흐름을 재확인해, 원래 의도가 문서/아키텍처 정리 → 노트북 실험 → `.py` 미러 diff → 통계 진단/리포트 → 자동화 분리였음을 복원했다.
- [x] `AGENTS.md` 최상단에 복원 규칙을 추가했다: `test/scripts` 1회성 파일 금지, 연구는 `test/models/*.ipynb + *.py`, workflow plumbing은 `.githooks`, 운영 entrypoint는 `pipelines`.
- [x] `test/README.md`에 test 디렉터리 정책을 추가했다.
- [x] `.githooks/check_repo_policy.py`를 추가하고 pre-commit에서 실행하도록 연결했다.
- [x] pre-commit 정책은 새 `test/scripts/*.py` 추가를 차단하고, `test/models/*.py`에 동명 `.ipynb`가 없으면 차단한다.
- [x] `test/scripts`에 잘못 넣었던 notebook mirror hook helper는 제거했고, `.githooks/sync_ipynb_mirrors.py`로 유지한다.
- [x] `__pycache__` 생성물을 제거했다.
- [x] `python -m py_compile .githooks\sync_ipynb_mirrors.py .githooks\check_repo_policy.py test\models\4_text_independent_variable_analysis.py`와 `python quantitative_trading\.githooks\check_repo_policy.py` 검증은 통과했다.
## 2026-06-09 교수님 메일 / 데이터마트 PDF 발송

- [x] 과거 흐름 기반 데이터마트 설명을 한국어 PDF로 먼저 렌더링해 깨짐을 확인했다.
- [x] PDF에는 과거 흐름 기반 데이터마트의 정의, 후보군(구간별/일정별/혼합형), 예시 시각화 3종을 넣었다.
- [x] 메일 본문에는 "코덱스로 작성해서 보내는 메일"이라는 점과, 현재 작업이 모델 내부 수정이 아니라 분석용 외생변수 선설계라는 점을 명시했다.
- [x] `test/scripts/send_historical_flow_datamart_professor_email_utf8.py`를 통해 PDF를 첨부해 교수님께 발송했다.
- [ ] 교수님 피드백을 받으면 구간 분할 기준과 혼합형 마트 스키마를 그에 맞춰 좁힌다.

## 2026-06-09 저장소 문서/임시 파일 정리 기록

- [x] 메일 발송용 임시 산출물 `test/results/historical_flow_datamart_professor_brief_20260609.{html,pdf,png}`를 제거했다.
- [x] 일회성 발송 스크립트 `test/scripts/send_historical_flow_datamart_professor_email_utf8.py`를 제거했다.
- [x] 중복 안내 문서 `test/skills.md`를 제거하고 루트 `skills.md` 중심으로 정리했다.
- [x] repo-local Meta Harness 참조 디렉터리 `.agents/skills/harness`, `.codex/skills/harness`를 제거했다.
- [x] Git stage를 비워 현재 정리 작업이 자동 커밋 대기 상태로 남지 않도록 했다.
- [ ] 다음 작업은 `/test` 분석 코드가 루트 패키지(`contexts.*`, `marts.*`)를 기준으로 이어질 수 있게 필요한 파일만 선별해 커밋 범위를 정리하는 것이다.

## 2026-06-09 문서 역할 분리 및 캐시 축소 기록

- [x] `AGENTS.md`를 영구 규칙 전용 문서로 재작성했다.
- [x] `skills.md`를 기술 철학, 분석 원칙, 보고서 기준 전용 문서로 재구성했다.
- [x] `README.md`를 사람용 프로젝트 개요, 구조, 실행 가이드 중심으로 재정리했다.
- [x] `conversation_l2_cache.md`에서 영구 규칙을 제거하고 최근 요청 5개만 남기도록 축소했다.
- [x] 세션 시작 시 읽을 우선순위를 `AGENTS.md` -> `process.md` -> `history.md` -> `conversation_l2_cache.md` -> `test/README.md`로 명시했다.
- [ ] 다음 작업은 현재 남아 있는 코드/문서 변경 범위를 검토해 커밋 단위를 정리하고, `/test` 연구 작업 재개 시 새 문서 체계를 기준으로 이어가는 것이다.

## 2026-06-09 테스트 스크립트 정리 및 L2 캐시 확장 기록

- [x] `conversation_l2_cache.md` 유지 원칙을 최근 핵심 요청 최대 20개로 확장했다.
- [x] `test/scripts/send_email.py`를 preset 기반 공용 UTF-8 메일 모듈로 통합했다.
- [x] `test/scripts/extract_notebook_images.py`만 남기고 노트북 후처리 유틸리티를 이미지 추출 용도로 단순화했다.
- [x] `test/scripts/build_exhaustive_notebook.py`, `build_notebook.py`, `enhanced_report_generator.py`, `notebook_to_md.py`, `reconstruct_test_env.py`를 제거했다.
- [x] `test/scripts/send_historical_flow_mart_email_utf8.py`, `send_independent_variables_report_email_utf8.py`, `send_text_context_update_email_utf8.py`를 제거했다.
- [x] `test/data/.gitkeep`를 추가하고 `test/README.md`에 `test/data`가 비어 있을 수 있는 이유와 남는 스크립트 역할을 기록했다.
- [x] `test/scripts/__pycache__`, `test/models/__pycache__`를 제거하고 남은 스크립트 `py_compile` 검증을 통과했다.
- [ ] 다음 작업은 남아 있는 코드/문서 변경 중 무엇을 함께 커밋할지 범위를 나누고, `/test` 연구 코드가 새 스크립트 체계만 참조하는지 추가 점검하는 것이다.
