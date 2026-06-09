# 퀀트 트레이딩 테스트 환경

이 디렉토리는 시계열 데이터 분석, 모델 비교, 결과 리포트 정리를 위한 실험 공간이다. 모든 설명은 한국어를 기본으로 하며, 코드 식별자와 명령은 원문을 유지한다.

## 디렉토리 구조

- **`models/`**: 실제 분석을 수행하는 Jupyter Notebook과 같은 이름의 `.py` 미러 파일
  - `1_time_series_test.ipynb`: 기초 시계열 분석과 RNN/LSTM 기준선
  - `2_time_series_advance_test.ipynb`: LSTM, GRU, TCN, Transformer, N-BEATS, mTAND, Mamba, Informer, Autoformer, PatchTST 등 고급 모델 비교
- **`results/`**: 분석 결과 리포트와 성능 지표
  - `analysis_report.md`: 모델 성능 비교와 최종 해석
- **`scripts/`**: 재사용 가능한 데이터 처리 및 변환 유틸리티
  - `send_email.py`: 재사용 가능한 UTF-8 메일 발송 모듈
  - `extract_notebook_images.py`: 노트북의 저장된 PNG 출력만 추출
- **`research_materials/`**: 분석에 사용한 논문, 요약, 참고 자료
- **`images/`**: 차트와 분석 결과 이미지

## 연구 흐름

1. 자료를 먼저 확인한다
   - `AGENTS.md`
   - `process.md`
   - `history.md`
   - `test/models/*.ipynb`
2. 실험을 노트북 우선으로 작성한다
   - `.ipynb`를 먼저 만들고 같은 이름의 `.py` 미러를 맞춘다
3. 필요한 경우 원격 커널에서 실행한다
   - 실제 결과 수치는 학교 서버나 승인된 원격 환경에서 산출한다
4. 결과를 문서화한다
   - 성능 수치, 변수 설명, 디버깅 이력, 해석을 한국어로 정리한다

## 최근 연구 연동

### 1. ArXiv MCP 서버

최신 논문과 연구 동향을 빠르게 확인할 때 사용한다. 예를 들어 `Codex CLI` 세션에서 `"2025 Mamba time series paper summaries"` 같은 질의를 던져 최신 연구를 수집한다.

### 2. Hugging Face Skills

모델, 데이터셋, 평가 도구를 확인할 때 사용한다. Hugging Face는 MCP 서버가 아니라 로컬 skill 세트로 취급한다.

### 3. 원격 커널 실행

대형 연구나 장시간 분석은 로컬이 아니라 승인된 원격 환경에서 수행한다. VS Code에서 `.py` 파일을 열고 Jupyter 서버를 원격 주소로 지정한 뒤 셀 단위로 실행한다.

### 4. 데이터 적재 복구

DuckDB 테이블이 비어 있거나 일부 데이터가 누락되면 재수집과 병합 흐름으로 복구한다. 이 방식은 로컬과 원격 환경 모두에서 같은 분석 경로를 유지하기 위해 필요하다.

## 주요 의존성

- **`torch`**: 신경망 모델 구현과 학습
- **`duckdb`**: 대용량 시계열 데이터 저장과 조회
- **`statsmodels`**: ADF 검정과 통계적 시계열 진단
- **`statsforecast`**: 고전 예측 모델 비교
- **`pyupbit`**: 업비트 시세 데이터 수집
- **`fastdtw` / `scipy`**: 시계열 형태 비교와 거리 계산
- **`optuna`**: 하이퍼파라미터 탐색

설치 예시:

```bash
pip install torch pandas numpy matplotlib duckdb scikit-learn optuna fastdtw scipy pyupbit statsmodels statsforecast
```

## 실행 흐름

1. 데이터 준비: 루트 `data/upbit_data.db`를 확인한다.
2. 모델 실행: `models/` 아래의 노트북을 열어 원격 커널에서 실험한다.
3. 최적화: 필요한 범위에서만 `Optuna`로 하이퍼파라미터를 조정한다.
4. 결과 정리: `results/`에 산출물을 모으고, 필요하면 `scripts/extract_notebook_images.py`로 이미지 출력만 추출한다.

## 2026-06-09 테스트 디렉토리 정책

- `test/models/`: 연구 실험 전용이다. 모든 실험은 `.ipynb` 원본과 같은 이름의 `.py` 미러를 함께 유지한다.
- `test/scripts/`: 재사용 가능한 공용 유틸리티만 둔다. 단발성 스크립트는 추가하지 않는다.
- 메일 스크립트는 `send_email.py` 한 모듈에서 preset 방식으로 재사용한다.
- 노트북 후처리 유틸리티는 이미지 추출 모듈 하나만 유지한다.
- 원천 DB와 공용 데이터 파일은 `test/`가 아니라 루트 `data/`에서 관리한다.
- `test/results/`, `test/images/`, `test/research_materials/`, `test/experiment_specs/`: 생성 산출물과 문서 자료 보관소다.
- Git hook 같은 워크플로우 플러밍은 `test/scripts/`가 아니라 `.githooks/`에 둔다.
- 운영 자동화는 `pipelines/`에 둔다.
- pre-commit 훅은 새 `test/scripts/*.py`와, 같은 이름의 `.ipynb`가 없는 `test/models/*.py`를 차단한다.
