# 📈 Quantitative Trading Test Environment

이 디렉토리는 가상자산 시계열 데이터 분석 및 모델 성능 검증을 위한 테스트 환경입니다.

## 📂 디렉토리 구조 및 역할

- **`data/`**: DuckDB(`upbit_data.db`) 및 가공된 시퀀스 데이터 저장.
- **`models/`**: 실제 분석을 수행하는 Jupyter Notebook 및 저장된 PyTorch 모델 파일.
  - `1_time_series_test.ipynb`: 기초 시계열 분석 및 RNN/LSTM 베이스라인.
  - `2_time_series_advance_test.ipynb`: Mamba, mTAND, TCN 등 고도화 모델 분석.
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
- **활용법**: `/ask "2025 Mamba time series paper summaries"`와 같이 요청하여 아카이브의 최신 연구를 즉각 반영합니다.
- **설치**: `mcpServers` 설정에 `@mcp/arxiv-server`를 등록하여 사용합니다.

### 2. Hugging Face Skills 활용 (모델 및 데이터셋 연동)
허깅페이스의 오픈소스 가중치나 데이터셋 메타데이터를 직접 조회합니다.
- **활용법**: `gemini extensions install https://github.com/huggingface/skills.git` 명령어로 스킬을 추가한 뒤, `Chronos`나 `TimesFM` 같은 파운데이션 모델 정보를 탐색합니다.

이를 통해 **단순히 코드를 짜는 것에 그치지 않고, 2025년 이후의 최신 학술 트렌드를 분석 프로세스에 직접 결합**할 수 있는 연구소급 환경을 구축했습니다.

### 3. 원격 주피터 서버 리소스 활용 (Remote Kernel for .py)
학교 서버와 같이 URL+Token으로만 접근 가능한 환경에서 `.py` 파일의 리소스를 사용하려면 **VS Code의 Interactive Window** 기능을 권장합니다.
- **설정 방법**:
  1. VS Code에서 `.py` 파일을 엽니다.
  2. 명령어 팔레트(`Ctrl+Shift+P`)에서 `Jupyter: Specify Jupyter Server for Connections`를 선택합니다.
  3. `Existing`을 선택하고 학교 서버의 **URL+Token** 주소를 입력합니다.
  4. `.py` 파일 상단의 `# %%` (코드 셀) 위에서 `Run Cell`을 클릭하면 원격 서버 리소스를 사용하여 코드가 실행됩니다.
- **장점**: `.ipynb`와 동일한 원격 리소스를 사용하면서도, 텍스트 기반인 `.py` 파일로 깔끔한 Git 커밋 이력(Diff) 관리가 가능합니다.

## 📦 의존성 패키지 (Dependencies)

본 테스트 환경의 코드를 실행하기 위해 필요한 패키지들입니다. 새로운 라이브러리 추가 시 이 섹션이 자동으로 업데이트됩니다.

- **기본 프레임워크**: `torch`, `pandas`, `numpy`, `matplotlib`, `duckdb`, `scikit-learn`
- **최적화 및 고급 지표**: `optuna`, `fastdtw`, `scipy`

### 설치 명령어
```bash
pip install torch pandas numpy matplotlib duckdb scikit-learn optuna fastdtw scipy
```

## 🚀 실행 프로세스


1. **데이터 준비**: `scripts/reconstruct_test_env.py`를 통해 DuckDB에 데이터 적재.
2. **모델링**: `models/` 내의 노트북을 실행하여 학습 및 성능 검증.
3. **최적화**: `Optuna`를 활용하여 하이퍼파라미터 튜닝 수행.
4. **결과 리포트**: 분석이 끝나면 결과를 `results/`에 정리하고 `analysis_report.md` 업데이트.
