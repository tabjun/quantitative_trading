# 에이전트 기술 셋 및 워크플로우 가이드 (Skills & AI Guidelines)

이 문서는 본 프로젝트에서 활용할 핵심 통계/분석 기술과, **Claude Code 및 Codex(Cursor)와 같은 AI 에이전트 도구를 200% 활용하기 위한 마크다운 세팅 및 운용 방안**을 담고 있습니다.

---

## 1. 다변량 퀀트 분석 핵심 기술 (Technical Skills)

코드를 생성할 때 AI는 아래의 라이브러리와 개념을 최우선적으로 활용해야 합니다.

- **Multivariate Time Series Analysis (다변량 시계열):**
  - 주가(Price, Volume) 트렌드와 함께, 외부 요인(금리, 감성 지수)을 결합하여 분석합니다.
  - *관련 라이브러리:* `pandas`, `statsmodels`, `scikit-learn`
- **Dynamic Time Warping (DTW):**
  - 유클리디안 거리의 한계를 극복하고 파동의 '형태(Shape)'적 유사성을 비교하여 과거 폭락장 사례를 매칭합니다.
  - *관련 라이브러리:* `fastdtw`, `scipy`
- **Sentiment & Macro NLP (문맥 감성 분석):**
  - 뉴스, 증권사 리포트, SNS 데이터의 텍스트를 임베딩 공간으로 매핑하여 특정 국면의 시장 분위기를 수치화합니다.
  - *관련 라이브러리:* `openai`, `sentence-transformers`
- **Defense-First Backtesting (방어 우선 백테스트):**
  - 교수님의 핵심 요구사항인 '최하방 방어'를 달성하기 위해, 수익 최적화보다는 MDD(Maximum Drawdown) 제어와 승률 방어에 초점을 맞춘 시뮬레이터를 작성합니다.

---

## 2. AI 코딩 에이전트 활용 및 마크다운 세팅 방안 (AI Tool Guidelines)

본 프로젝트는 AI 에이전트(Claude Code, Cursor/Codex 등)가 주도적으로 개발을 진행합니다. 에이전트의 환각(Hallucination)을 막고 컨텍스트를 유지하기 위해 아래의 환경 세팅을 강력히 권장합니다.

### 2.1. 프로젝트 룰 파일 구성 (`.cursorrules` / `.clauderules` 추천)
에이전트 도구가 디렉토리를 열었을 때 자동으로 전역 컨텍스트를 인지할 수 있도록 프로젝트 루트에 룰 파일을 생성하는 것이 좋습니다.
- **룰 파일 내용 추천:**
  - *"항상 작업을 시작하기 전에 `history.md`와 `process.md`를 읽어 현재 Step을 파악하라."*
  - *"의존성 설치 및 실행은 반드시 `uv` 패키지 매니저(`uv run`, `uv add`)를 사용하라."*
  - *"코드를 변경하기 전, 문서는 `npx -y @chrisryugj/kordoc ./materials/...`로 파싱하여 참고하라."*
  - *"모든 매매 로직의 기준점은 '수익 극대화'가 아닌 '하방 방어'다."*

### 2.2. 컨텍스트 주입 프롬프트 (세션 초기화 시)
Claude Code를 터미널에서 실행하거나 Cursor 채팅 창을 열었을 때, 첫 질문으로 아래 프롬프트를 복사하여 붙여넣으면 매우 효과적입니다.

> "현재 이 프로젝트는 다변량 데이터와 AI를 결합한 퀀트 트레이딩 '하방 방어' 모델을 구축 중이야. 
> 네가 현재 어떤 상황인지 파악할 수 있도록 `README.md`, `process.md`, `history.md`를 우선적으로 읽어줘. 
> 다 읽었다면 `history.md`의 'Next Step'에 해당하는 작업을 바로 시작할 구체적인 계획과 실행 코드를 제시해줘."

### 2.3. 토큰 효율성 및 모듈화 전략
- **세션 분리:** 코드를 수십 번 핑퐁하며 수정하다 보면 에이전트의 컨텍스트 윈도우(기억력)가 오염됩니다. `process.md`에 정의된 Step 하나가 끝나면 과감히 세션을 초기화(`/clear` 또는 새 탭 열기) 하십시오.
- **결과 캐싱:** 작업이 끝날 때마다 에이전트에게 **"지금까지 완성한 코드의 핵심 구조와 다음 세션을 위한 인계 사항을 `history.md`에 기록해줘"**라고 명령하여 영구 기억 공간을 업데이트합니다.

---

## 3. DevOps 및 확장 인프라 가이드 (Future CI/CD & Automation)

향후 시스템이 커지고 실시간 프로덕션 서비스로 확장될 경우를 대비한 DevOps 기술 스택 활용 방법입니다. AI 에이전트는 향후 아래의 시스템들과 연동하는 코드를 작성하게 될 수 있습니다.

### 3.1. n8n (워크플로우 자동화)
- **사용처:** 파이프라인 오케스트레이션 및 알림 자동화
- **사용 방법 (예시):**
  1. n8n을 Docker로 서버에 띄웁니다 (`docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n`).
  2. **Trigger Node:** 매일 오후 3시 30분(장 마감 후) 실행되도록 Cron 노드를 설정합니다.
  3. **Action Node:** Python 스크립트(`uv run main.py`)를 서버에서 실행하는 HTTP Request 또는 Execute Command 노드를 연결합니다.
  4. **Output Node:** 생성된 최종 분석 리포트(JSON/Markdown)를 Slack이나 Telegram 노드로 전송하여 모바일로 받아볼 수 있도록 구성합니다.

### 3.2. Docker & Kubernetes (컨테이너 오케스트레이션)
- **사용처:** 다중 에이전트 독립 배포 및 리소스 스케일링
- **사용 방법 (예시):**
  1. **Dockerize:** `uv` 환경이 세팅된 베이스 이미지를 기반으로, `Situation Analyzer`, `Similarity Matcher` 등 각각의 에이전트 모듈을 독립된 `Dockerfile`로 빌드합니다.
  2. **K8s Deployment:** `deployment.yaml`을 작성하여 쿠버네티스 클러스터에 배포합니다. 
  3. 시장 폭락 등의 이슈로 분석 요청이 몰릴 때, 패턴 매칭 엔진(Pod)의 개수를 자동으로 늘리도록 HPA(Horizontal Pod Autoscaler) 정책을 적용합니다.

### 3.3. Sonatype Nexus (아티팩트 관리)
- **사용처:** 프라이빗 Docker Registry 및 모델 가중치(Pickle), 내부 Python 패키지 캐싱
- **사용 방법 (예시):**
  1. 사내망 또는 클라우드에 Nexus를 구축하고 Docker Hosted Repository를 생성합니다.
  2. GitHub Actions 등의 CI 파이프라인에서 분석 엔진 이미지를 빌드한 뒤 `docker push <nexus-url>/quant-engine:v1.0` 명령어로 업로드합니다.
  3. K8s 클러스터는 서비스 배포 시, 외부 퍼블릭 망(Docker Hub)이 아닌 안전하고 빠른 내부 Nexus 저장소에서 이미지를 가져와(`imagePullSecrets` 활용) 실행하도록 구성합니다.
