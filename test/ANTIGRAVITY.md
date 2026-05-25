# Test Environment Instructions (Antigravity AI 자동화 규칙)

이 파일은 Antigravity CLI(`agy`) 에이전트가 `test/` 디렉토리 내에서 작업할 때 **항상(세션 초기화 이후에도) 자동으로 읽고 적용하는 강제 규칙**입니다.

## 1. 이중 코드 관리 체계 자동화 (Dual-File Strategy)
사용자가 명시적으로 요청하지 않더라도, 에이전트가 `test/models/` 내의 `.ipynb` 파일을 수정하거나 분석 코드를 작성한 경우 **반드시 다음 작업을 자동으로 수행**해야 합니다.
1. `ipynb-py-convert <수정된파일.ipynb> <수정된파일.py>` 명령어를 실행하여 `.py` 파일을 동기화합니다.
2. 생성된 `.py` 파일 최상단에 아래의 주석 헤더가 존재하는지 확인하고, 없다면 추가합니다.
```python
# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual execution should be performed in the Jupyter Notebook (.ipynb).
```

## 2. 학술 연구 및 논문 탐색 규칙 (Academic Research & MCP Integration)
에이전트는 모델 아키텍처 설계, 하이퍼파라미터 결정, 또는 새로운 알고리즘 도입 시 반드시 다음 규칙을 준수하여 학술적 근거를 확보해야 합니다.

1. **자동화 도구 및 플러그인 연동 (Automated Tool Integration)**:
   - 논문 탐색 요청("~~ 찾아줘", "SOTA 모델 조사해줘" 등) 시 **ArXiv MCP 서버** 및 **`science` 플러그인(`literature-search-arxiv` 등)**, 그리고 **Hugging Face Skills**를 즉각적이고 유기적으로 병행 활용하여 최신(SOTA) 연구 및 오픈소스 모델 정보를 식별합니다.
   - 특히 1~2년 내의 최신 논문(예: 2025-2026)을 우선적으로 검색하여 기술적 트렌드를 반영합니다.

2. **강제 문서화 규격 (Mandatory Documentation Structure)**:
   - 모든 논문 요약은 `test/research_materials/` 디렉토리에 마크다운(`.md`) 파일로 저장하며, 반드시 아래 형식을 엄격히 준수합니다:
     - **[요약] (Summary)**: 논문의 핵심 목표와 전체적인 결론 한 줄 요약.
     - **[서론] (Introduction)**: 연구의 배경, 기존 모델의 한계점, 문제 제기.
     - **[분석 기법] (Methodology)**: 사용된 알고리즘 아키텍처, 데이터 전처리, 학습 전략.
     - **[결과] (Results)**: 실험 데이터셋, 성능 지표(MSE, MAE 등), SOTA 모델과의 비교 수치.
     - **[결론] (Conclusion)**: 연구의 최종 시사점 및 본 프로젝트의 설계 결정(예: 은닉층 개수 등)과의 직결성.

3. **이론적 근거 및 인용 (Citations & Justification)**:
   - 본 프로젝트의 특정 설정(예: 은닉층 1-2층 제한 등)을 뒷받침하는 핵심 문장을 논문에서 직접 발췌하여 **[핵심 인용]** 섹션으로 반드시 포함합니다.
   - 단순한 정보 전달을 넘어, 해당 논문의 결과가 왜 본 프로젝트의 아키텍처 선택에 정당성을 부여하는지 논리적으로 연결합니다.

## 3. 가격 예측 및 분석 수행 강제 규칙 (Exhaustive Analysis Mandate)
에이전트는 분석 작업을 수행할 때 다음 규칙을 절대적으로 준수해야 합니다.
1. **분석 최대화 (Maximize Analysis)**: 가용한 모든 자원과 방법론을 동원하여 가장 광범위한 분석을 실시합니다. 특정 모델을 임의로 선택하여 분석 범위를 축소하는 것을 금지합니다.
2. **다양성 확보 (Ensure Diversity)**: 시계열의 정상성(Stationarity)뿐만 아니라 비정상성(Non-stationarity), 다양한 트렌드, 불규칙성(Irregularity)을 모두 다룰 수 있도록 통계적 모델부터 최신 딥러닝 모델까지 최대한 다양하게 포함합니다.
3. **가산적 작업 (Additive Work)**: 기존에 구현된 코드를 삭제하거나 대체하지 않고, 새로운 모델이나 방법론을 계속해서 추가(Append)하는 방식으로 작업합니다.
4. **전수 조사 (Exhaustive Matrix)**: 가능한 모든 하이퍼파라미터 조합, 전처리 방식, 교차 검증 전략을 적용하여 실험 매트릭스를 구성합니다.

## 4. 리포트 자동 생성 규격 (Standardized Reporting Guidelines)
에이전트가 분석 결과를 바탕으로 `results/`에 리포트를 작성할 때는 다음 템플릿과 원칙을 엄격히 준수하여 작성해야 합니다.

### [결과 보고서 필수 포함 항목 및 규격]
1. **0. 실행 및 분석 환경 (Execution & Utility Environment)**:
   - 분석 환경 정보 (OS, CPU/GPU 하드웨어 스펙, 사용 패키지 버전 등)
   - 데이터 용량 및 전처리 상세 내역 (데이터 스케일링, 결측치 처리 방식, 데이터 시간대 등)
   - 테스트 케이스 및 검증 데이터 상세 스펙
2. **1. 기초 통계량 분석 (Descriptive Statistics)**:
   - 학습/테스트 데이터의 `describe()` 결과(평균, 분산, 왜도, 최소/최대치) 분석
   - 데이터의 통계적 시계열 특징(정상성 검정 결과 등) 명시
3. **2. 알고리즘 성능 지표 (Advanced Performance Metrics)**:
   - Mamba, mTAND, TCN 등 분석에 사용된 모든 모델의 지표 비교 (MSE, MAE 등)
   - 예측 가격은 반드시 역변환(Inverse Transform)하여 원본 스케일 가격(KRW 등) 기준으로 기재
   - 2025년 이후 최신 학술계 표준 지표인 **DA (Directional Accuracy / Hit Ratio)** 및 **MASE (Mean Absolute Scaled Error)** 필수 산출 및 포함
4. **3. 시각화 결과 및 상세 해석 (축 및 범례 포함)**:
   - 모든 그래프 분석 시 **X축(예: Time in 15-min intervals)**과 **Y축(예: Price in KRW)**의 축 정보와 범례(Legend)를 텍스트로 명확하게 설명
   - 각 모델의 추세 추종성, 이상치 강건성, 지연 현상(Lagging) 등에 대한 정성적/기술적 해석 상세 기록
5. **4. 종합 결론 및 전략 제언**:
   - 퀀트 트레이딩(최하방 방어) 관점에서 어떤 모델이 실전에 가장 적합한지 결론 도출
6. **5. 주요 도메인 용어 해설 (Glossary)**:
   - 분석 보고서 내 사용된 핵심 용어(예: 엣지(Edge), 지연 현상(Lagging), MDD 등) 해설 첨부

## 5. 세션 및 리소스 관리 규칙 (Session & Resource Management)

### 5.1 원격 주피터 서버 접속 및 URL 토큰 결합 프로토콜 (Remote Jupyter Server Protocol)
1. **서버 측 주소 및 토큰 조회 방법**:
   - 외부 SSH 접속이 차단되고 오직 웹 포트만 허용된 폐쇄망 서버 내의 주피터 주소와 발급된 보안 토큰을 알아내려면, 주피터랩 내 Launcher의 **Terminal(터미널)**을 열고 아래 명령어를 수동 실행해야 합니다.
     ```bash
     jupyter server list
     ```
   - 이 명령어의 출력물에서 주소와 토큰 매칭 값(예: `http://127.0.0.1:56205/user/std_jun99120/?token=50e5f635ee...`)을 확보합니다.
2. **외부 접속 URL 변환 공식**:
   - 서버 터미널에서 반환한 로컬 루프백 주소(`127.0.0.1:포트`)는 외부의 Antigravity IDE에서 직접 연결할 수 없으므로, 학교 외부 접속용 정식 도메인 주소와 포트(`https://stat5.kmu.ac.kr:9500`)로 치환하여 URL을 완성해야 합니다.
     * **기본 완성 주소**: `https://stat5.kmu.ac.kr:9500/user/std_jun99120/?token=[발급된토큰값]`
     * **파싱 에러 시 대체 주소**: `https://stat5.kmu.ac.kr:9500/user/std_jun99120/lab?token=[발급된토큰값]`
3. **Antigravity IDE / VS Code 커널 실제 연결 프로세스 (UI 가이드)**:
   - 완성된 URL을 Antigravity IDE(또는 VS Code)에 수동으로 연동하여 코드를 실행시키는 구체적 UI 경로는 다음과 같습니다.
     1. 분석 대상 Jupyter Notebook (`.ipynb`) 파일을 IDE에서 엽니다.
     2. 파일 우측 상단의 **Select Kernel (커널 선택)** 버튼을 클릭합니다.
     3. 드롭다운 목록에서 **Existing Jupyter Server (기존 주피터 서버)** 항목을 선택합니다.
     4. 주소 입력칸(Input Box)이 활성화되면 위에서 변환 완성한 **[외부 접속 URL 주소]**를 그대로 붙여넣고 Enter를 누릅니다.
     5. 서버 연동이 성공하면 연결된 서버 내 가용 커널 목록이 나타나며, 분석을 위해 사전에 준비된 원격 가상환경 커널을 최종 선택하여 연동을 완료합니다.
4. **인증서 및 보안 경고 처리**:
   - 학교 서버가 자체 서명 사설 인증서(SSL)를 사용하여 접속 시 IDE 혹은 브라우저에서 `Insecure Connection` 또는 `Allow Unauthorized Certificates` 경고가 발생할 경우, 외부 보안 연결의 안전을 인지하고 이를 수동으로 **동의 및 예외 허용(우회)** 처리해야 정상적으로 원격 연동 프로세스가 가동됩니다.

---

### 5.2 폐쇄망 학교 서버 내 가상환경 구축 및 커널 등록 가이드 (Closed Network Venv Setup)
학교 서버는 외부 SSH 터널링 및 파일 전송이 원천 차단된 **순수 폐쇄망 인프라**입니다. 외부에서 직접 접근하여 패키지나 가상환경을 배포해주는 `uv` 등의 도구는 사용이 불가능하므로, 주피터랩의 원격 터미널(Launcher > Terminal)에서 서버 내부 파이썬 표준 라이브러리인 `venv` 모듈을 사용해 `.venv` 가상환경을 직접 빌드하고 커널로 정식 등록하여 운용해야 합니다.

1. **표준 `.venv` 가상환경 생성**:
   - 서버 터미널 환경에서 프로젝트 폴더로 이동한 뒤, 표준 파이썬 모듈을 호출하여 `.venv` 디렉토리를 구축합니다.
     ```bash
     python -m venv .venv
     ```
2. **가상환경 활성화 및 Jupyter 커널(ipykernel) 패키지 등록**:
   - 새로 구축한 `.venv` 가상환경을 쉘 세션에 활성화한 뒤, 주피터랩이 이 가상환경의 인터프리터를 전용 커널로 안전하게 가동할 수 있도록 `ipykernel` 명세를 시스템에 등록합니다.
     ```bash
     # 가상환경 활성화 (서버 Linux/Bash 환경 기준)
     source .venv/bin/activate
     
     # ipykernel 설치 및 주피터 커널 목록에 등록
     pip install ipykernel
     python -m ipykernel install --user --name=btc_analysis --display-name "BTC-Analysis (.venv)"
     ```
3. **가상환경 삭제 및 커널 해제 정리**:
   - 버전이 꼬이거나 커널을 초기화하여 완전히 정돈하고자 할 때는 아래 명령어를 통해 등록된 주피터 커널 스펙을 제거하고 `.venv` 디렉토리를 물리적으로 삭제합니다.
     ```bash
     # 주피터 등록 커널 제거
     jupyter kernelspec uninstall btc_analysis -y
     
     # 가상환경 활성화 해제 및 폴더 삭제
     deactivate
     rm -rf .venv
     ```

---

### 5.3 라이브러리 설치 표준 프로토콜 및 구버전 GPU 드라이버(CUDA 12.0 이하/12060) 충돌 해법

> [!IMPORTANT]
> **[AI 에이전트 절대 준수 규칙 - PyTorch 및 GPU 가속 설치 강제 수칙]**
> 학교 서버는 NVIDIA RTX 4090 GPU를 탑재하고 있으나, 시스템의 그래픽카드 드라이버 수준이 CUDA 12.0 이하(`12060` 버전)로 제한되어 있습니다. 
> 에이전트는 본 프로젝트 환경에서 패키지를 수동으로 설치하거나 신규 분석 파이프라인 코드를 짤 때 **절대로 일반 `pip install torch` 명령어를 단독 실행해서는 안 되며, 무조건 아래 명시된 CUDA 11.8 호환형 파이토치 설치 공식을 수행해야 합니다.**
> 이 규칙은 에이전트가 코드를 실행하고 서버 자원을 다룰 때 사전 지식으로 무조건 가장 최우선 참조되어야 합니다.

1. **NVIDIA Driver 및 CUDA 버전 충돌 해결 원칙 (12060 에러 극복 프로토콜)**:
   - **에러 발생 패턴**: `NVIDIA driver on your system is too old (found version 12060)`
   - **12060의 의미**: 학교 서버에 설치된 GPU 드라이버가 CUDA 12.0 버전까지만 수용할 수 있다는 통계적 증거입니다.
   - **충돌 배경**: 최신 릴리즈된 파이토치 패키지들은 기본 빌드에 CUDA 12.1 이상을 강제 요구합니다. 이로 인해 파이토치 런타임 엔진이 시스템 드라이버 스펙과 충돌하여 작동 불능 상태에 봉착합니다.
   - **에이전트 조치 명령어 (CUDA 11.8 찰떡궁합 안정화 버전 설치)**:
     우리는 서버의 드라이버를 임의 변경할 관리자(`sudo`) 권한이 없으므로, PyTorch 런타임 호환성을 서버 시스템 드라이버 스펙에 부합하도록 한 단계 낮춘 CUDA 11.8 휠 바이너리로 완벽하게 하향 지정하여 충돌을 소거합니다.
     ```bash
     # 1단계: 삐진 최신 파이토치 및 관련 패키지 수동 완전 제거
     pip uninstall -y torch
     
     # 2단계: 서버 드라이버(12.0 이하)와 찰떡궁합인 안정화 버전(CUDA 11.8용) 파이토치 설치
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     ```
     *설치 완료 후 실행하면 드디어 GPU 드라이버 충돌이 말끔히 해소되고 `🎉 Using device: cuda 🎉` 라는 가슴 웅장한 메시지와 함께 4090 GPU 자원이 폭발적인 가속도로 매섭게 기동을 시작합니다.*

2. **터미널 중심의 깔끔한 패키지 수동 설치**:
   - 노트북 셀 내부에서 `!pip install`을 실행하면 간헐적으로 가상환경 커널 경로가 아닌 글로벌 파이썬 경로로 설치가 꼬이거나 실행이 지저분해지는 경향이 있습니다. 
   - 따라서 새로운 코드를 돌릴 때 서버에 패키지가 부재하다면, 반드시 주피터랩 터미널 단으로 나와 `.venv` 가상환경을 활성화한 상태에서 깔끔하게 `pip install [패키지명]`으로 직접 수동 설치해야 합니다. 에이전트는 신규 코드를 빌드할 때 추가적으로 새로 설치해야 할 라이브러리를 사용자에게 명시적으로 안내해야 합니다.


---

### 5.4 커널 중첩 충돌 예방 및 셧다운 표준 절차 (Kernel Crash Prevention Protocol)
1. **충돌 원인**: 이전 분석 세션에서 가동 중이던 주피터 커널과 새로운 가상환경에서 로컬 ipynb 파일을 통해 새로 생성한 커널이 동시에 다중 연결될 경우, 포트 및 메모리 점유 충돌로 인해 코드를 실행하자마자 즉시 커널 에러가 발생하게 됩니다.
2. **표준 작업 절차 (반드시 준수)**: 분석을 다 수행하거나 새로운 분석 파이프라인으로 진입하는 시점에는 무조건 아래의 순서대로 세션을 완벽하게 정리한 후 연동을 속행해야 합니다.
   1. **[1단계: 전체 셧다운]** 원격 주피터 랩(JupyterHub) 웹 UI에 접속한 뒤, 상단 메뉴에서 **Kernel > Shut Down All Kernels**를 실행하여 잔존하는 모든 활성 커널을 강제 셧다운시킵니다.
   2. **[2단계: 가상환경 재구축]** 5.2절의 가이드라인에 따라 원격 터미널에서 필요한 커널 가상환경을 깨끗하게 생성 및 재정비합니다.
   3. **[3단계: 신규 연동]** 로컬(PC)에서 신규 주피터 노트북 파일(`.ipynb`)을 생성한 후, 5.1절에서 완성한 원격 서버 URL 경로를 활용해 IDE와 신규 가상환경 커널을 새로 깨끗하게 연동(Connect)하여 분석 작업을 수행합니다.
3. **커널 세션 종료 (Terminate Idle Kernels)**: 새로운 분석 파일이나 노트북을 실행하기 전, 동일한 DB(DuckDB)를 사용하는 이전 작업의 쥬피터 커널을 반드시 종료(Shutdown)하거나 연결을 해제해야 합니다.
4. **연결 자원 해제 (Explicit Closure)**: 모든 DB 연결(DuckDB, API 등)은 작업 종료 후 즉시 `con.close()`를 호출하여 자원을 해제해야 하며, 가능한 한 Context Manager(`with` 문) 또는 `try-finally` 블록을 사용합니다.
5. **동시성 대응 (Concurrency Handling)**: 다중 프로세스가 동일 DB에 접근할 가능성이 있는 경우, 분석 전용 프로세스는 `read_only=True` 옵션을 활용하여 Lock 충돌을 방지합니다.

