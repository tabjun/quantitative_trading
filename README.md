# AI 퀀트 트레이딩 연구: 다변량 분석 기반 최하방 방어 모델

## 1. 연구 개요 (Research Overview)
본 연구는 전통적인 통계학적 시계열 모델의 한계를 극복하고, 거대언어모델(LLM)과 다변량 분석(Multivariate Analysis)을 결합하여 '**최하방 지지선 방어**'에 특화된 퀀트 매매 전략을 탐색합니다. 
단순히 수익률을 극대화하는 것이 아니라, 역사적 패턴 유사도(DTW)와 거시적 감성 분석(Sentiment)을 결합하여 하락장에서의 손실을 원천 차단하는 **'절대로 잃지 않는' 모델** 구축이 핵심 목표입니다.

## 2. 핵심 연구 방향 (Strategic Direction)
- **Univariate to Multivariate:** 주가 데이터(단변량)에 의존하지 않고 뉴스, 전쟁, 매크로 지표, SNS 반응 등 외부 변수를 포함한 다변량 분석을 수행합니다.
- **Pattern & Sentiment Matcher:** 현재의 주가 모양과 비슷한 과거 시점을 찾고, 당시의 '뉴스 맥락(감성)'까지 비교하여 현재 흐름을 예측합니다.
- **Data Mart 기반 분석:** 매번 API를 호출하는 비효율을 제거하기 위해, 섹터별/이슈별로 정제된 데이터 마트를 구축(DB/AWS)하여 즉각적인 백테스트 환경을 조성합니다.
- **AI-Agentic Workflow:** 연구의 전 과정(데이터 수집, 코딩, 분석, 리포트)을 Claude Code, Codex 등의 AI 에이전트와 완벽히 통제된 환경에서 협업하여 수행합니다.

## 3. 도메인 지식 자료 (Domain Knowledge Materials)

본 프로젝트는 알고리즘 트레이딩 시스템 구축을 위한 도메인 지식 강화를 위해 [WikiDocs: Cherry Quant](https://wikidocs.net/148475)의 내용을 정리하여 `materials/quant_and_stock.md`에 반영하였습니다. 이 자료는 AI 에이전트와 연구원이 프로젝트 수행 중 참고해야 할 필수 지식 베이스이자 연구 방향의 이탈을 막는 가드레일 역할을 합니다.

### 3.1. 활용 방법 (Usage for Agents & AI Tools)
- **컨텍스트 주입:** AI 에이전트(Claude Code 등)는 분석이나 코드 작성 전 반드시 `materials/quant_and_stock.md`를 읽고 퀀트 트레이딩의 기본 프로세스(전략 인식 -> 백테스트 -> 실행 -> 리스크 관리)를 준수해야 합니다.
- **용어 일관성:** 프로젝트 내의 변수 명명, 리포트 작성 시 `018. 주식 용어 정리` 섹션에 정의된 표준 용어 및 은어를 사용하여 의사소통의 명확성을 유지합니다.
- **지식 베이스 쿼리:** 특정 퀀트 전략(예: 에드워드 소프의 랜덤워크, 로보어드바이저 유형 등)에 대한 이론적 근거가 필요할 때 해당 문서를 우선적으로 참조합니다.

### 3.2. 가드레일 역할 (Guardrails)
- **연구 방향성 유지:** 본 프로젝트의 목표인 '최하방 방어' 전략 수립 시, `002. 알고리즘 트레이딩 프로세스`와 `015. 월가아재의 허와실`에 언급된 '과최적화 방지', '리스크 관리' 원칙을 위반하지 않도록 체크리스트로 활용합니다.
- **검증 기준 제공:** 백테스팅 결과 분석 시 샤프 지수(Sharpe Ratio), 최대 낙폭(MDD) 등 `002` 섹션에서 정의된 계량적 지표를 반드시 포함하도록 강제합니다.

## 4. 시스템 아키텍처 (Architecture)
아래는 본 프로젝트의 데이터 수집부터 분석, 시각화까지의 전체 데이터 파이프라인입니다.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryBorderColor': '#333333', 'lineColor': '#666666', 'fontFamily': 'sans-serif'}}}%%
graph TD
    classDef data fill:#b3e5fc,stroke:#039be5,stroke-width:2px,rx:5px,ry:5px;
    classDef proc fill:#ffe0b2,stroke:#fbc02d,stroke-width:2px,rx:5px,ry:5px;
    classDef engine fill:#c8e6c9,stroke:#43a047,stroke-width:2px,rx:5px,ry:5px;
    classDef ui fill:#e1bee7,stroke:#d81b60,stroke-width:2px,rx:5px,ry:5px;

    L1["1. Data Acquisition Layer (수집)<br/>- MTS / Upbit API (실시간 거래가)<br/>- News / SNS / Macro (거시 데이터)"]:::data
    L2["2. Data Engineering Layer (전처리/마트 구축)<br/>- 과거 데이터 섹터별 심층 분석<br/>- 정제된 섹터/이슈 DB"]:::proc
    L3["3. Multivariate Analysis Layer (다변량 분석)<br/>- Situation Analyzer<br/>- Similarity Matcher<br/>- 최하방 방어 백테스트"]:::engine
    L4["4. Execution Layer (결과 도출 및 실행)<br/>- Visual Dashboard<br/>- AI 매매 규칙 생성<br/>- 퀀트 자동 매매 실행"]:::ui

    L1 -- "시장/거시 데이터 전달 (분석 데이터 활용)" --> L2
    L1 -- "타겟 종목 실시간 거래가 전달" --> L3
    L2 -- "과거 유사 패턴 통계치 제공 (독립변수 활용)" --> L3
    L3 -- "최종 투자 판단 및 규칙" --> L4
```

### 🖼️ System Architecture Image
![System Architecture](materials/quant_architecture.png)

## 4. 환경 설정 및 실행 방법 (Setup & Execution)

### 4.1. 레포지토리 클론 및 의존성 설치
본 프로젝트는 속도와 의존성 관리에 최적화된 `uv` 패키지 매니저를 기반으로 작성되었습니다.

```bash
# 1. 저장소 클론
git clone <repository_url>
cd stock

# 2. uv를 활용한 가상환경 생성 및 종속성 동기화
uv sync

# 3. 메인 분석 스크립트 실행
uv run main.py
```
### 4.2. Kordoc MCP 활용 (문서 파싱)
연구 자료나 참고 문서를 AI 에이전트가 완벽하게 숙지할 수 있도록 `kordoc`을 활용합니다. 에이전트 세션 시작 시 아래 명령어를 통해 컨텍스트를 주입하세요.

```bash
# npx를 통한 kordoc 실행 (문서 파싱 및 요약)
npx -y @chrisryugj/kordoc ./materials/research_overview.docx
```
*에이전트는 위 명령의 출력을 읽어 프로젝트의 세부 목적을 잊지 않고 분석 코드를 작성해야 합니다.*

### 4.3. grill-me 스킬 설치 및 사용 방법

본 프로젝트는 기획 및 분석의 엣지 케이스를 발굴하고 완성도를 높이기 위해 `grill-me` 스킬을 활용합니다.

#### 1. 스킬 설치
터미널에 다음 명령어를 입력하여 스킬 패키지 설치를 시작합니다.

```bash
npx skills@latest add mattpocock/skills
```

설치 명령어를 실행하면 상호작용 형태의 선택 창이 나타납니다.
1. **스킬 선택:** 방향키를 이용해 `grill-me`와 `grill-with-docs`를 찾아 스페이스바로 체크하고 엔터를 누릅니다.
2. **에이전트 선택:** 대상 에이전트를 묻는 메시지가 나오면, 목록에서 `Gemini CLI`를 찾아 선택합니다. (Universal 그룹에 기본 포함되어 있습니다.)
3. **설치 범위(Installation scope):** `Global`을 선택하여 모든 프로젝트에서 사용할 수 있도록 설정하는 것을 권장합니다.

#### 2. 사용 방법
설치가 완료되면 제미나이 CLI 프롬프트에서 슬래시 명령어를 입력하여 스킬을 호출합니다.

*   **/grill-me [주제]:** 특정 모듈이나 로직에 대해 제미나이와 인터뷰 모드로 진입하여 구체적인 기준점과 엣지 케이스를 논의합니다.
*   **/grill-with-docs [주제]:** 작업 경로의 파일들을 먼저 분석한 뒤 문맥에 맞는 날카로운 질문을 던지며, 합의된 내용을 바탕으로 `CONTEXT.md` 및 `ADR` 문서를 자동 생성합니다.

### 4.4. 테스트 및 분석 환경 (Test & Analysis Environment)
`test/` 디렉토리는 엄격한 학술적/실무적 검증을 위해 아래와 같이 구조화되어 있습니다.
- **`models/`**: 실제 모델 학습 및 성능 분석을 수행하는 Jupyter Notebook 파일들.
- **`results/`**: 분석 결과 리포트(`analysis_report.md`) 및 최종 성능 지표 저장.
- **`research_materials/`**: Mamba, mTAND 등 분석에 사용된 최신 논문 및 이론적 배경 정리.
- **`skills.md`**: 테스트 환경 전용 분석 가이드라인 및 운영 수칙.

---

## 5. AI 에이전트(Claude Code / Codex) 협업 가이드

이 레포지토리는 AI 에이전트가 컨텍스트를 잃지 않고 작업을 이어갈 수 있도록 설계되었습니다. 에이전트 도구 실행 시 다음 수칙을 따르십시오.

1. **초기화:** Claude Code나 Cursor(Codex)를 열었을 때 가장 먼저 `history.md`와 `process.md`를 읽도록 지시합니다.
   - *Prompt ex: "현재 진행 상황을 파악하기 위해 history.md와 process.md를 읽고 다음 스텝을 제안해줘."*
2. **도구 최적화:** `skills.md`에 작성된 가이드라인을 바탕으로 에이전트가 어떤 라이브러리(e.g., fastdtw, transformers)를 사용할지 제한하고 최적화합니다.
3. **세션 종료:** 단일 작업(Step)이 끝나면 반드시 `history.md`에 결과를 요약 기록하게 하여 다음 세션으로 컨텍스트를 안전하게 인계합니다.

## 6. 향후 확장 및 배포 파이프라인 (Future Scalability)

본 프로젝트는 연구 단계를 넘어 실제 프로덕션(Production) 환경의 실시간 분석 시스템으로 확장될 수 있도록 다음과 같은 DevOps 인프라 연동을 염두에 두고 있습니다.

### 6.1. Workflow Automation (n8n)
- **목적:** 데이터 수집, 파이프라인 트리거, 결과 알림 전송을 자동화합니다.
- **연동 방안:** MTS API에서 데이터를 주기적으로 수집하는 스케줄러(Cron)나 분석된 AI 리포트를 Telegram/Slack으로 전송하는 파이프라인을 n8n의 노드 형태로 시각화하고 자동화합니다.

### 6.2. Containerization & Orchestration (Docker & Kubernetes)
- **목적:** 다수의 에이전트(상황 분석기, 패턴 매칭기 등)를 독립적인 마이크로서비스(MSA)로 분리하고 무중단 운영을 보장합니다.
- **연동 방안:** `uv` 기반의 파이썬 앱을 최적화된 Docker 이미지로 빌드합니다. 이후 시장 변동성이 심해져 분석 요청이 폭증할 경우, Kubernetes(K8s)의 HPA(Horizontal Pod Autoscaler)를 통해 매칭 엔진 Pod을 자동으로 스케일 아웃합니다.

### 6.3. Artifact Management (Sonatype Nexus)
- **목적:** 프라이빗 Docker 이미지, 학습이 완료된 딥러닝 모델 파일, 파이썬 패키지 등을 안전하게 저장하고 버전 관리합니다.
- **연동 방안:** CI/CD 파이프라인(GitHub Actions 등)을 통해 빌드된 Docker 이미지를 Nexus Repository에 Push하고, Kubernetes 클러스터 배포 시 외부 퍼블릭 망이 아닌 내부 Nexus 망에서 이미지를 Pull하여 보안과 속도를 향상시킵니다.
