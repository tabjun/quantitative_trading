# 에이전트 기술 셋 및 워크플로우 가이드 (Skills & AI Guidelines)

본 문서는 프로젝트에서 활용할 핵심 통계/분석 기술과, **Codex CLI 에이전트 도구를 200% 활용하기 위한 마크다운 세팅, 전사 규칙 및 세션 초기화 대응 방안**을 담고 있습니다.

---

## 0. 실행 주체 분리 규칙 (Execution Actor Boundary)

이 레포지토리는 다른 연구원/운영자가 재현 실행할 수 있도록 자동화 코드, CLI 엔트리포인트, 실행 명령 예시를 보존합니다. 다만 Codex는 사용자의 개인 연구 세션에서 대용량/장시간 연구 수행 코드(딥러닝 학습, 노트북 분석 실행, 백테스트, 분석 리포트 산출용 실험)를 로컬 터미널로 직접 실행하지 않습니다. 실제 학습/분석 실행은 학교 서버 커널, CI/스케줄러, 또는 사용자가 지정한 원격 실행 환경에서 수행합니다.

- **레포 보존 범위**: 자동화 실행 코드, `uv run ...` 명령 예시, n8n/Cron/CI/Docker/Kubernetes 연동 설계, 서버 커널 실행 절차.
- **Codex 수행 범위**: 코드 작성, 정적 코드 리뷰, 설계 검토, 간단한 로컬 계산, 문법/컴파일 점검, import 확인, 작은 단위 테스트, 서버 커널 실행 절차 문서화, 연구용 `.ipynb`와 동명 `.py` 미러 파일 작성/동기화.
- **Codex 금지 범위**: 개인 연구 세션의 로컬 터미널 또는 `.venv`에서 `uv run main.py`, `uv run pipelines/simulate_and_send.py` 같은 장시간 분석/학습/백테스트 파이프라인 실행, 노트북 연구 결과 산출 실행, 딥러닝 학습, 대규모 추론, 백테스트/분석 결과 생성용 실행.
- **예외**: 사용자가 해당 세션에서 명시적으로 로컬 실행을 승인한 경우에만 제한적으로 수행합니다.

---

## 1. 다변량 퀀트 분석 핵심 기술 (Technical Skills)

코드를 생성하거나 분석을 진행할 때 에이전트는 아래의 라이브러리와 개념을 최우선적으로 활용해야 합니다.

- **Multivariate Time Series Analysis (다변량 시계열):**
  - 주가(Price, Volume) 트렌드와 함께, 외부 요인(뉴스 감성 지수, 매크로 경제 지표)을 결합하여 분석합니다.
  - *관련 라이브러리:* `pandas`, `statsmodels`, `scikit-learn`
- **Dynamic Time Warping (DTW):**
  - 유클리디안 거리의 한계를 극복하고 파동의 '형태(Shape)'적 유사성을 비교하여 과거 폭락장 사례를 매칭합니다.
  - *관련 라이브러리:* `fastdtw`, `scipy`
- **Sentiment & Macro NLP (문맥 감성 분석):**
  - 뉴스, 증권사 리포트, SNS 데이터의 텍스트를 임베딩 공간으로 매핑하여 특정 국면의 시장 분위기를 수치화합니다.
  - *관련 라이브러리:* `sentence-transformers` (로컬 CPU/GPU 리소스 또는 로컬 Hugging Face Skills 활용)
- **Defense-First Backtesting (방어 우선 백테스트):**
  - 교수님의 핵심 요구사항인 '최하방 방어'를 달성하기 위해, 수익 최적화보다는 MDD(Maximum Drawdown) 제어와 승률 방어에 초점을 맞춘 시뮬레이터를 작성합니다.

---

## 2. Codex CLI (Codex CLI) 기반의 전사적 규칙 표준 (Corporate Guidelines)

본 프로젝트는 `Codex CLI` 에이전트의 오작동 및 환각을 방지하고, 다중 연구원 간의 정밀한 세션 초기화와 인계를 보장하기 위해 다음과 같은 엄격한 표준을 적용합니다.

### 2.1. 최우선 전사 룰북 (AGENTS.md)
- 프로젝트 루트의 `AGENTS.md` 파일은 `Codex CLI`가 기동될 때마다 자동으로 임포트되어 내재적 규칙으로 적용됩니다.
- 에이전트에게 수동으로 분석 기준을 일일이 상기시키지 않아도, 시작 시 자동으로 `history.md`와 `process.md`를 스캔하고 아래의 학술 및 분석 보고 표준 규격을 일관되게 고수하도록 설계되었습니다.

### 2.2. 최신 학술 탐색 연동 규격
- SOTA(State-of-the-Art) 모델 아키텍처나 파라미터 튜닝 시, `Codex CLI`에 내장/연동된 **ArXiv MCP 서버** 및 로컬 `.agents/skills/`에 구성된 **Hugging Face Skills**를 적극적으로 활용합니다.
- 탐색된 모든 논문 요약은 `test/research_materials/` 디렉토리에 마크다운(`.md`) 파일로 저장하되, 반드시 **요약 -> 서론 -> 분석 기법 -> 결과 -> 결론(및 [핵심 인용] 구문)**의 표준 5단계 형식을 엄격하게 적용하여 문서화해야 합니다.

### 2.3. 전사 분석 결과 보고 규격 (Standardized Reporting Guidelines)
에이전트가 분석 결과를 바탕으로 `results/` 등에 리포트를 작성할 때는 반드시 다음 템플릿과 원칙을 예외 없이 100% 준수해야 합니다.
1. **0. 실행 및 분석 환경 (Execution & Utility Environment)**
   - OS 및 CPU/GPU 하드웨어 사양, 핵심 패키지 버전.
   - 데이터 총 용량, 전처리 과정(스케일링 및 보간 방식) 및 테스트 케이스 스펙.
2. **1. 기초 통계량 분석 (Descriptive Statistics)**
   - 분석 데이터의 `describe()` 결괏값 기술 및 시계열의 통계적/정상성 검정 특징 명시.
3. **2. 알고리즘 성능 지표 (Advanced Performance Metrics)**
   - 비교 대상 모델(Mamba, mTAND, TCN 등) 전체의 오차 지표 기록.
   - **원본 KRW 스케일 표기:** 예측가는 반드시 전처리 이전의 원본 가격 스케일로 **역변환(Inverse Transform)**하여 기재.
   - **DA 및 MASE 필수 산출:** 2025년 이후 학술계 표준인 **DA (Directional Accuracy)** 및 **MASE (Mean Absolute Scaled Error)** 필수 포함.
4. **3. 시각화 결과 및 상세 해석 (축 및 범례 텍스트 명시)**
   - 그래프 해석 시 **X축(예: Time in 15-min intervals)**과 **Y축(예: Price in KRW)**의 설정과 범례를 텍스트로도 완벽히 매칭 설명.
   - 추세 추종성과 지연 현상(Lagging) 해소 강도에 대해 기술적으로 논증.
5. **4. 종합 결론 및 전략 제언**
   - 최하방 방어 관점에서 실전에 적합한 알고리즘과 매매 임계값 제안.
6. **5. 주요 도메인 용어 해설 (Glossary)**
   - 보고서에 등장하는 전문 용어(예: 엣지(Edge), 라깅(Lagging), MDD 등)의 설명 첨부.

---

## 3. 주기적 세션 초기화 및 full-auto 활용 가이드

금융 분석 및 대규모 시계열 모델 훈련 프로세스 특성상, 토큰 한계 도달이나 캐시 오염을 막기 위해 연구원은 주기적으로 세션을 초기화해야 합니다. `Codex CLI` 환경에서는 이를 다음과 같은 팁으로 손쉽게 극복할 수 있습니다.

### 🚀 Tip 1: full-auto (자동 승인) 모드 구동
매 파일 읽기/쓰기나 명령어 실행 시 일일이 "허용"을 마우스 클릭이나 엔터로 입력하는 번거로움을 생략하려면, 터미널 실행 시 **`--dangerously-skip-permissions`** 플래그를 추가하십시오.
```bash
Codex CLI --dangerously-skip-permissions "학습 데이터 정제하고 SQLite 적재해줘"
```
> [!WARNING]
> 이 옵션은 모든 시스템 제어와 도구 작동을 자동 승인하므로, 신뢰할 수 있는 퀀트 트레이딩 프로젝트 폴더 내부에서만 실행하는 것을 권장합니다.

### 🚀 Tip 2: 세션 리셋 후 원클릭 맥락 복원 및 실행
주기적으로 세션을 초기화(`/clear` 또는 터미널 재실행)한 뒤, 이전 작업을 완벽히 계승하여 다음 Step을 곧바로 실행하게 만드는 매직 프롬프트 팁입니다.
```bash
Codex CLI --dangerously-skip-permissions "history.md와 process.md를 읽고 현재 Next Step 작업을 바로 실행해줘"
```
이 한 줄의 명령만 실행하면, `Codex CLI`는 `AGENTS.md`에 명시된 규칙에 따라 `history.md`와 `process.md`를 스캔하고, 현재 어느 Phase의 어떤 Step에 머무르고 있는지 파악한 뒤, ToDo 리스트의 다음 미완료 항목을 기계적으로 자동 이행하기 시작합니다.

---

## 4. DevOps 및 확장 인프라 가이드 (Future CI/CD & Automation)

향후 시스템이 실시간 프로덕션 매매 서비스로 확장될 경우, AI 에이전트는 아래의 DevOps 기술 스택과 연동할 수 있습니다.

### 4.1. n8n (워크플로우 자동화)
- **사용처:** 파이프라인 오케스트레이션 및 리포트 Telegram/Slack 자동 발송.
- **연동 예시:** 매일 오후 3시 30분(장 마감 후) 실행되는 Cron 노드를 설정하여, 학교 서버/원격 실행 환경에서 `uv run main.py`를 호출하고 그 결과 생성된 표준 리포트 Markdown을 연구원의 Slack으로 즉시 발송합니다. 이 자동화 코드는 레포에 보존해야 하며, 금지되는 것은 Codex가 개인 연구 세션에서 로컬 터미널로 직접 분석 실행을 수행하는 행위입니다.

### 4.2. Docker & Kubernetes (MSA 배포)
- **사용처:** 에이전트 독립 배포 및 리소스 스케일링.
- **연동 예시:** `uv` 기반의 시계열 분석 엔진을 컨테이너로 빌드하고, 시장 폭락장 등 분석 요청이 급증하는 시점에 Kubernetes의 HPA(Horizontal Pod Autoscaler)를 통해 패턴 매칭 Pod 개수를 자동으로 증가시킵니다.

### 4.3. Sonatype Nexus (아티팩트 및 가중치 관리)
- **사용처:** 프라이빗 Docker Registry 및 모델 가중치(Pickle, GGUF), 내부 패키지 캐싱.
- **연동 예시:** 학습이 완료된 딥러닝 모델 가중치나 비공개 패키지 등을 사내 Nexus 저장소에 Push하여 안전하게 보존하고, 프로덕션 배포 시 보안성 높은 내부 Nexus 망에서 이를 Pull하여 빠르게 서비스를 구동합니다.
