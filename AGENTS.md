# AGENTS.md - Codex Custom Instructions & Working Agreements

Welcome to the Quantitative Trading & 시계열 (Time Series) Prediction Project workspace. As an AI Agent (Codex), you must strictly adhere to the following rules, standards, and workflow procedures in this repository.

---

## 1. 세션 초기화 및 기억 복원 (Session Persistence & Context Recovery)

> [!IMPORTANT]
> **최우선 탐색 의무 (First-Scan Directive)**
> 새로운 대화 세션이 시작되거나 세션 초기화/리셋이 감지되면, 다른 어떤 프롬프트 지시보다도 **가장 먼저 프로젝트 루트의 `history.md`와 `process.md` 파일을 정밀 스캔**하여 현재 진행 상태를 완벽히 동기화하십시오.

- **맥락 파악**: 두 문서를 통해 현재 수행 중인 '연구 단계(Phase)', '최종 완료된 태스크', 그리고 '즉시 진행해야 하는 Next Step'을 파악하여 인지 오차 없이 작업을 계승해야 합니다.
- **작업 종료 전 상태 갱신**: 단일 태스크나 Step을 마친 후 작업을 종료하기 전, 다음 세션으로의 완벽한 컨텍스트 인계를 위해 **반드시** 아래 작업을 자동 수행하십시오:
  1. `process.md`를 열어 완료된 Step을 업데이트하고 다음 목표를 수립합니다.
  2. `history.md`에 오늘 날짜와 작업 이력 테이블을 추가하고, **"세션 인계를 위한 핵심 요약"**란을 최신 상태로 수정하십시오.

---

## 2. 학술 탐색 및 SOTA 연동 규칙 (Academic Research & MCP Integration)

- **도구 및 플러그인 연동**:
  - 논문 조사 또는 SOTA(State-of-the-Art) 모델 연구 지시를 받으면, 수동 웹 서칭 대신 구성된 **ArXiv MCP 서버** 및 로컬 `.agents/skills/` 내의 **Hugging Face Skills**를 적극 활용하여 정밀하게 자료를 탐색하십시오.
  - 2025~2026년 이후의 시계열 및 퀀트 최신 학술 동향을 우선 반영합니다.
- **학술 자료 문서화 규격 (Standardized Research Paper Format)**:
  - 수집되거나 인용된 논문 내용은 `test/research_materials/` 디렉토리에 마크다운(`.md`) 파일로 저장해야 하며, 반드시 아래의 **표준 5단계 포맷**을 엄격히 적용하십시오:
    1. **[요약] (Summary)**: 논문의 핵심 아키텍처와 결론 1줄 요약.
    2. **[서론] (Introduction)**: 기존 모델의 기술적 한계 및 연구 배경.
    3. **[분석 기법] (Methodology)**: 알고리즘 아키텍처, 데이터 전처리 흐름, 학습 파라미터 상세 기재.
    4. **[결과] (Results)**: SOTA 모델 대비 지표(MSE, MAE 등)의 구체적 향상도.
    5. **[결론 및 설계 결정] (Conclusion)**: 해당 연구가 우리 프로젝트의 설계(예: 특정 레이어 층수, 노드 개수 결정 등)에 기여하는 명확한 이론적 근거 및 **[핵심 인용]** 구문 수록.

---

## 3. 분석 및 예측 보고서 표준 규격 (Exhaustive Analysis & Reporting Standards)

분석 결과 보고서(프로젝트 루트 및 `test/results/` 폴더 내 마크다운)는 반드시 아래 **6대 핵심 기준**을 완벽하게 만족해야 하며, 내용을 절대로 임의 축소하지 마십시오.

- **0. 실행 및 분석 환경 (Execution & Utility Environment)**
  - 분석 환경의 메타데이터(OS, CPU/GPU, 주요 라이브러리 버전 등) 명시.
  - 데이터셋 용량, 시간 범위(예: 15분 단위 3년 데이터), 전처리 기법(데이터 스케일링, 결측치 보간 등)과 Train/Val/Test 분할 스펙 기술.
- **1. 기초 통계량 분석 (Descriptive Statistics)**
  - 분석 대상 시계열 데이터의 `describe()` 결괏값(평균, 분산, 왜도, 최소/최대치 등) 해석.
  - 정상성(Stationarity) 검정 등 통계적 시계열 특징 명시.
- **2. 알고리즘 성능 지표 (Advanced Performance Metrics)**
  - 비교 및 사용한 모든 모델(Mamba, mTAND, TCN, PatchTST, Autoformer 등)의 오차 지표 기록.
  - **스케일 역변환 필수**: 예측 가격이나 지표는 전처리용 normalized scale에 머무르지 않고, 반드시 **역변환(Inverse Transform)**을 적용하여 **원본 가격 스케일(KRW 등)** 기준으로 표기하십시오.
  - **학술 지표 필수 산출**: 2025년 이후 학술계 표준인 **DA (Directional Accuracy / Hit Ratio)** 및 **MASE (Mean Absolute Scaled Error)** 필수 포함. (MASE가 1보다 작은지 검증하여 단순 Persistence 모델 대비 우월성을 수치로 증명할 것)
- **3. 시각화 결과 및 상세 해석 (축 및 범례 텍스트 명시)**
  - 모든 그래프 설명 시, 반드시 **X축(예: Time in 15-min intervals)**과 **Y축(예: Price in KRW)**의 축 정보 및 범례(Legend) 텍스트를 논리적으로 매치하여 설명하십시오.
  - 추세 추종 능력, 지연 현상(Lagging) 해소 강도에 대해 기술적으로 정밀 분석.
- **4. 종합 결론 및 전략 제언**
  - "절대 잃지 않는다"는 **최하방 방어(MDD 최소화)** 퀀트 관점에서 최선의 알고리즘 및 매매 임계점(Thresholds) 설정 제안.
- **5. 주요 도메인 용어 해설 (Glossary)**
  - 보고서 내의 핵심 도메인 용어(예: 엣지(Edge), 라깅(Lagging), MDD 등)를 비전문가 눈높이에 맞게 해설.
- **6. 개발 과정, 디버깅 이력 및 심층 탐구 (Development & Debugging Logs)**
  - 단순 수치 요약을 넘어, **이전에 비해 실제 변경 및 수정한 코드 스니펫(Diff)**을 반드시 리포트에 삽입하십시오.
  - 직면했던 **기술적 오류(예: PyTorch 차원 불일치 등)의 진단 경로와 완전한 코드 해결책** 기록.
  - **시계열 지연 매핑(Lag-1 Shift) 메커니즘 심층 분석**: epoch 1 수준에서 훈련 손실이 수직 낙하하며 고착되던 현상은 '지연 매핑(Lag-1 Shift)' 편법 가중치 쏠림(이전 시점 가격을 다음 시점 가격으로 단순 카피 예측하여 loss를 낮추는 Trapping)이 원인임을 밝히고, 이를 극복하기 위해 수행한 PreprocessingPipeline의 지연 극복 전처리 설계와 그에 따른 점진적 손실 하강 안정화 결과를 정성/정량적으로 상세 기술하십시오.

---

## 4. 모델 설계 철학 (Model Design Philosophy)

> [!TIP]
> **"Shallow but Wide" 원칙**
> 금융 시계열 데이터의 극심한 노이즈와 과적합(Overfitting) 방지를 위해, 딥러닝/시계열 예측 신경망 레이어는 **1~2층의 얕은 구조(Shallow)**로 설계하되, 각 레이어의 노드 수(Width)는 **64~128개**로 비교적 넓게 가져가는 기조를 엄격하게 유지하십시오.

---

## 5. Codex 실행 및 테스트 명령 가이드

본 프로젝트에서는 패키지 관리 및 빠른 툴 기동을 위해 **`uv`** 환경을 사용합니다. 작업을 시작하거나 테스트할 때 다음 명령 규격을 사용하십시오.

- **스크립트 기동 및 테스트**:
  - `uv run main.py`
  - `uv run analyzer.py`
- **Jupyter Notebook 변환 및 점검**:
  - 만약 `.ipynb` 파일을 갱신하는 경우, 변경점 추적이 용이하도록 동명의 `.py` 파일로 미러링(Mirroring)하여 동기화할 것을 강력히 권장합니다.
- **SQLite DB 확인**:
  - SQLite 클라이언트나 Python 스크립트를 사용하여 `upbit_data.db`에 있는 15분봉 등의 테이블 적재 현황을 직접 쿼리하여 검증할 수 있습니다.

---

## 6. Git 자동 커밋 및 GitHub 마크다운 렌더링 경로 전송 규칙 (Git Auto-Push & Rendered Link Delivery)

> [!IMPORTANT]
> **보고서 전달 자동화 필수 전제조건 (Delivery Aesthetics & Accessibility)**
> 보고서 마크다운(`.md`) 파일을 이메일로 자동 전송할 때, 단순 날것(Raw Text) 형식의 첨부 방식은 열람 가독성을 저해하므로 금지합니다. 에이전트는 반드시 보고서를 원격 저장소에 자동 반영하고, GitHub 고유의 미려한 스타일시트 렌더링 경로를 추출하여 이메일 본문에 연동해야 합니다.

- **자동 Git 라이프사이클 집행**:
  - 보고서(`analysis_report.md` 등) 또는 모의 투자 결과 파일이 생성/갱신되면, 이메일 발송 스크립트는 백그라운드에서 `git add`, `git commit -m "..."`, `git push origin stock` 명령어 세트를 자동으로 집행하여 GitHub 원격 `stock` 브랜치에 코드를 즉각 푸시해야 합니다.
- **GitHub 렌더링 URL 구성 및 이메일 본문 연동**:
  - 푸시 완료 후, GitHub 원격 주소 체계를 판독하여 해당 브랜치(`stock`) 하위의 파일 경로 주소(예: `https://github.com/tabjun/personal_ai_project/blob/stock/quantitative_trading/analysis_report.md`)를 동적으로 구성하십시오.
  - 구성된 깃허브 실시간 마크다운 렌더링 링크를 이메일 본문(Email Body) 상단에 명확하게 삽입하고, 해당 보고서 내의 핵심 개선점(수수료 차감 내역, AI 차트 판독 논리 등)을 메일 텍스트에 상세히 요약 소개하여 보고서 접근성과 열람 편의성을 극대화하십시오.
