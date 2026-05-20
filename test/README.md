# 📈 Quantitative Trading Test Environment

이 디렉토리는 가상자산 시계열 데이터 분석 및 모델 성능 검증을 위한 테스트 환경입니다.

## 📂 디렉토리 구조 및 역할

- **`data/`**: DuckDB(`upbit_data.db`) 및 가공된 시퀀스 데이터 저장.
- **`models/`**: 실제 분석을 수행하는 Jupyter Notebook 및 저장된 PyTorch 모델 파일.
  - `1_time_series_test.ipynb`: 기초 시계열 분석 및 RNN/LSTM 베이스라인.
  - `2_time_series_advance_test.ipynb`: **총 10종의 고도화 모델** (LSTM, GRU, TCN, Transformer, N-BEATS, mTAND, Mamba, Informer, Autoformer, PatchTST) 전수 분석.
- **`results/`**: 분석 결과 리포트 및 성능 지표 데이터.
  - `analysis_report.md`: 모델 성능 분석 및 전략 제언 보고서.
- **`scripts/`**: 데이터 수집, 변환, 알림 전송 등을 수행하는 유틸리티 스크립트.
  - `create_advance_nb.py`: 고도화 분석용 노트북 생성 스크립트.
  - `notebook_to_md.py`: 분석 결과를 마크다운 리포트로 변환.
- **`research_materials/`**: 분석에 사용된 알고리즘의 논문 요약 및 이론적 배경.
- **`images/`**: 시각화된 차트 및 분석 결과 이미지.

## 🔬 실시간 학술 트렌드 추적 및 연구소급 환경 (Academic Trend Tracking)

본 테스트 환경은 단순한 코드 구현을 넘어, 최신 SOTA(State-of-the-Art) 모델의 문헌을 실시간으로 추적하고 반영할 수 있도록 설계되었습니다.

### 1. Arxiv MCP Server 활용 (문헌 검색)
최신 논문의 초록을 읽고 알고리즘의 원리를 실시간으로 파악합니다.
- **활용법**: `agy` 세션에서 `"2025 Mamba time series paper summaries"`와 같이 논문 탐색을 요청하여 아카이브의 최신 연구를 즉각 반영합니다.
- **설치**: `.agents/mcp_config.json`에 `arxiv` MCP 서버를 등록하여 `agy`와 연동해 사용합니다.

### 2. Hugging Face Skills 활용 (모델 및 데이터셋 연동)
허깅페이스의 오픈소스 가중치나 데이터셋 메타데이터를 직접 조회합니다.
- **활용법**: `.agents/skills/` 디렉터리에 Hugging Face Skills 리포지토리의 스킬들을 위치시켜 `agy`에서 네이티브 스킬로 직접 활용하며, `Chronos`나 `TimesFM` 같은 파운데이션 모델 정보를 탐색합니다.

이를 통해 **단순히 코드를 짜는 것에 그치지 않고, 2025년 이후의 최신 학술 트렌드를 분석 프로세스에 직접 결합**할 수 있는 연구소급 환경을 구축했습니다.

### 3. 원격 주피터 서버 리소스 활용 (Remote Kernel for .py)
학교 서버와 같이 URL+Token으로만 접근 가능한 환경에서 `.py` 파일의 리소스를 사용하려면 **VS Code의 Interactive Window** 기능을 권장합니다.
- **설정 방법**:
  1. VS Code에서 `.py` 파일을 엽니다.
  2. 명령어 팔레트(`Ctrl+Shift+P`)에서 `Jupyter: Specify Jupyter Server for Connections`를 선택합니다.
  3. `Existing`을 선택하고 학교 서버의 **URL+Token** 주소를 입력합니다.
  4. `.py` 파일 상단의 `# %%` (코드 셀) 위에서 `Run Cell`을 클릭하면 원격 서버 리소스를 사용하여 코드가 실행됩니다.
- **장점**: `.ipynb`와 동일한 원격 리소스를 사용하면서도, 텍스트 기반인 `.py` 파일로 깔끔한 Git 커밋 이력(Diff) 관리가 가능합니다.

### 4. 자가 치유 데이터 파이프라인 (Self-Healing DuckDB Data Loading)
본 시스템은 데이터 부재로 인한 분석 중단을 방지하기 위해 **실시간 자동 수집 로직**을 내장하고 있습니다.
- **작동 원리**: DuckDB 연결 시 테이블(`btc_15m_advance`)이 없거나 데이터가 부족한 경우, `pyupbit`를 통해 실시간 시장 데이터를 즉시 수집하여 DB를 재구축합니다.
- **장점**: 어떤 환경(로컬, 학교 서버 등)에서도 별도의 데이터 세팅 없이 즉시 분석 실행이 가능하며, 항상 최신 데이터를 기반으로 연구를 진행할 수 있습니다.

## 📦 의존성 패키지 (Dependencies)

본 테스트 환경의 전수 분석(Exhaustive Analysis)을 위해 추가된 주요 라이브러리 설명입니다.

- **`torch`**: 15종 이상의 최신 시계열 딥러닝 모델(Mamba, Transformer 등) 구현 및 학습.
- **`duckdb`**: 대규모 시계열 데이터의 고속 저장 및 쿼리 (자가 치유 파이프라인의 핵심).
- **`statsmodels`**: ADF Test(정상성 검정) 및 STL 분해(트렌드/계절성 분석) 등 통계적 검증.
- **`statsforecast`**: 고성능 통계 모델(AutoARIMA 등)을 통한 딥러닝 베이스라인 비교.
- **`pyupbit`**: 실제 시장 데이터 실시간 수집 및 DB 동기화.
- **`fastdtw` / `scipy`**: 시계열 파동의 형태적 유사도(Dynamic Time Warping) 측정.
- **`optuna`**: 하이퍼파라미터 전수 조사를 위한 자동 최적화 프레임워크.

### 설치 명령어
```bash
pip install torch pandas numpy matplotlib duckdb scikit-learn optuna fastdtw scipy pyupbit statsmodels statsforecast
```

## 🚀 실행 프로세스


1. **데이터 준비**: `scripts/reconstruct_test_env.py`를 통해 DuckDB에 데이터 적재.
2. **모델링**: `models/` 내의 노트북을 실행하여 학습 및 성능 검증.
3. **최적화**: `Optuna`를 활용하여 하이퍼파라미터 튜닝 수행.
4. **결과 리포트**: 분석이 끝나면 결과를 `results/`에 정리하고 `analysis_report.md` 업데이트.
