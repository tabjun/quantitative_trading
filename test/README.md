# 연구 실험 공간 안내

`test/`는 운영 프레임워크가 아니라 연구 실험, 노트북, 결과물, 메모를 두는 공간입니다.

## 1. 폴더 역할

- `models/`: 연구용 `.ipynb` 원본과 동명 `.py` 미러
- `results/`: Markdown 보고서, 요약표, 선택적 CSV
- `research_materials/`: 문헌 메모, 실험 설계, 교수님 공유 브리프
- `images/`: 노트북 출력 그림과 보고서 삽입용 이미지
- `scripts/`: 재사용 가능한 보조 유틸리티만 보관

## 2. 공통 규칙

- 새 실험은 `test/models/*.ipynb`부터 만든다.
- 같은 이름의 `.py` 미러를 함께 유지한다.
- 후속 연구가 기존 번호 실험의 의미를 바꾸면, 기존 파일을 덮어쓰지 않고 새 번호 실험(`5 -> 6 -> 7`)으로 분리한다.
- 무거운 연구 실행은 학교 서버 커널이나 승인된 원격 환경에서 한다.
- 루트 프레임워크로 올릴 로직은 `analysis/`, `contexts/`, `marts/`, `pipelines/`로 옮긴다.
- 노트북을 파일별로 하나씩 실행한 뒤에는 서버의 `Kernel -> Shut Down All Kernels`로 이전 커널을 완전히 종료하고 다음 파일을 연다.
- 원격 Jupyter 서버에 다시 붙을 때는 서버 `IP:port`와 `token`으로 접속하며, 커널 상태가 꼬이면 재연결보다 기존 커널 종료를 먼저 한다.

## 3. 현재 주요 실험

- `1_time_series_test.ipynb`: 기초 시계열 모델 비교
- `2_time_series_advance_test.ipynb`: 고급 시계열 아키텍처 비교
- `3_time_series_multi_ticker_test.ipynb`: KRW 다종목 실험
- `4_text_independent_variable_analysis.ipynb`: 텍스트 독립변수 결합 실험
- `5_optimization_diagnostics_test.ipynb`: 최적화 경로와 shortcut collapse 진단
- `6_optimization_stabilization_test.ipynb`: target, normalization, loss, model selection 안정화 실험

## 4. 4번부터 6번까지의 연구 흐름

- `4번`은 텍스트와 외생변수가 가격 예측에 실제 정보 가치를 주는지 보는 실험이다.
- `5번`은 objective, target, architecture가 직전가 복사나 0 수익률 같은 쉬운 해로 붕괴하는지 보는 진단 실험이다.
- `6번`은 독립변수와 데이터마트를 본격적으로 붙이기 전에 target, normalization, loss, model selection 기준을 안정화하는 실험이다.
- 문헌 기반 논문화 방향과 후속 알고리즘 후보는 `test/research_materials/forecasting_methodology_literature_review_20260613.md`를 본다.
- 세부 해석은 각 노트북의 결과 셀과 `test/results/*.md` 보고서를 우선 본다.
- 설계 메모와 참고문헌은 `test/experiment_specs/`에 둔다.

## 5. 실행과 산출물

- 기본 실행 예시는 각 연구 노트북 안의 셀과 `test/models/*.py`에 둔다.
- 결과 해석의 1차 원본은 `ipynb` 출력 셀이다.
- `test/scripts/extract_notebook_images.py`는 노트북 출력 그림 추출용 보조 유틸리티다.
- `results/`는 보고서와 필요 시 CSV를 둔다.
- 새 보고서는 이전 보고서를 참고하더라도 알고리즘/손실함수/정규화/지표/그래프 읽는 법을 다시 적는 독립 문서로 작성한다.
- `test/README.md`에는 폴더 안내만 유지하고, 긴 연구 서술은 노트북/보고서로 보낸다.

## 6. 서버 환경 구성

이 저장소는 반복되는 서버 환경 구축을 스크립트로 고정한다.

- `test/scripts/bootstrap_uv_313.sh`: `uv`로 Python 3.13 환경을 새로 만들고, `uv sync`, CUDA 체크, PyTorch wheel 설치, Jupyter kernel 등록까지 한 번에 수행한다.
- `test/scripts/bootstrap_uv_312.sh`: `uv`로 Python 3.12 환경을 새로 만들고 같은 순서로 정리한다.
- `test/scripts/bootstrap_venv_313.sh`: 서버에 `python3.13`이 실제로 있을 때만 plain `venv` 방식으로 같은 작업을 수행한다.
- `test/scripts/bootstrap_venv_312.sh`: 서버에 `python3.12`가 실제로 있을 때만 plain `venv` 방식으로 같은 작업을 수행한다.
- 모든 스크립트는 고정 `.venv`를 덮어쓰지 않고 `.venvs/<env_name>` 형태로 새 환경을 만든다.
- 기본 환경 이름은 날짜/시간이 붙은 형태로 자동 생성되며, 실행 전에 기존 env 디렉터리와 Jupyter kernel 목록도 같이 출력한다.

실행 예시는 다음처럼 쓴다.

```bash
bash test/scripts/bootstrap_uv_312.sh
bash test/scripts/bootstrap_uv_313.sh
bash test/scripts/bootstrap_venv_312.sh
bash test/scripts/bootstrap_venv_313.sh
```

```bash
# 이름을 직접 주고 싶을 때
ENV_NAME=my_quant_313 KERNEL_NAME=my_quant_313 bash test/scripts/bootstrap_uv_313.sh
```

- 학교 서버에서는 먼저 `git pull --rebase origin <branch>`로 최신 브랜치를 맞추고, 그다음 환경 스크립트를 실행한다.
- 환경이나 노트북을 수정한 뒤에는 `git add`, `git commit`, `git push origin <branch>` 순서로 남긴다.
- 커널을 파일별로 바꿔 가며 실행할 때는 항상 `Kernel -> Shut Down All Kernels`를 먼저 수행한다.
- JupyterLab에서 커널이 안 보이면, 스크립트가 만든 `Python 3.12 (Quant Stat) [env_name]` 또는 `Python 3.13 (Quant Stat) [env_name]` 커널을 다시 선택한다.
- 스크립트 실행이 끝나면 해당 env 경로, 커널 이름, 제거 명령(`rm -rf ...`, `jupyter kernelspec uninstall ...`)도 같이 출력된다.

## 7. 참고 문서

- [루트 README](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/README.md)
- [AGENTS.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/AGENTS.md)
- [process.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/process.md)
- [history.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/history.md)
