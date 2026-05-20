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
1. **커널 세션 종료 (Terminate Idle Kernels)**: 새로운 분석 파일이나 노트북을 실행하기 전, 동일한 DB(DuckDB)를 사용하는 이전 작업의 쥬피터 커널을 반드시 종료(Shutdown)하거나 연결을 해제해야 합니다.
2. **연결 자원 해제 (Explicit Closure)**: 모든 DB 연결(DuckDB, API 등)은 작업 종료 후 즉시 `con.close()`를 호출하여 자원을 해제해야 하며, 가능한 한 Context Manager(`with` 문) 또는 `try-finally` 블록을 사용합니다.
3. **동시성 대응 (Concurrency Handling)**: 다중 프로세스가 동일 DB에 접근할 가능성이 있는 경우, 분석 전용 프로세스는 `read_only=True` 옵션을 활용하여 Lock 충돌을 방지합니다.
