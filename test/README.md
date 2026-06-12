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
- 무거운 연구 실행은 학교 서버 커널이나 승인된 원격 환경에서 한다.
- 루트 프레임워크로 올릴 로직은 `analysis/`, `contexts/`, `marts/`, `pipelines/`로 옮긴다.

## 3. 현재 주요 실험

- `1_time_series_test.ipynb`: 기초 시계열 모델 비교
- `2_time_series_advance_test.ipynb`: 고급 시계열 아키텍처 비교
- `3_time_series_multi_ticker_test.ipynb`: KRW 다종목 실험
- `4_text_independent_variable_analysis.ipynb`: 텍스트 독립변수 결합 실험
- `5_optimization_diagnostics_test.ipynb`: 최적화 경로와 shortcut collapse 진단

## 4. 4번과 5번 실험 요약

- `4번`은 텍스트가 가격 예측에 실제로 도움이 되는지 보는 실험이다.
- `5번`은 objective, target, architecture가 쉬운 해로 붕괴하는지 보는 진단 실험이다.
- 세부 해석은 각 노트북의 결과 셀과 `test/results/*.md` 보고서를 우선 본다.
- 설계 메모와 참고문헌은 `test/experiment_specs/`에 둔다.

## 5. 실행과 산출물

- 기본 실행 예시는 각 연구 노트북 안의 셀과 `test/models/*.py`에 둔다.
- 결과 해석의 1차 원본은 `ipynb` 출력 셀이다.
- `test/scripts/extract_notebook_images.py`는 노트북 출력 그림 추출용 보조 유틸리티다.
- `results/`는 보고서와 필요 시 CSV를 둔다.
- `test/README.md`에는 폴더 안내만 유지하고, 긴 연구 서술은 노트북/보고서로 보낸다.

## 6. 참고 문서

- [루트 README](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/README.md)
- [AGENTS.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/AGENTS.md)
- [process.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/process.md)
- [history.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/history.md)
