# 연구 실험 공간 안내

`test/`는 실제 운영 프레임워크가 아니라 연구 실험, 노트북, 문헌 메모, 결과물을 관리하는 공간입니다. 루트 패키지와 역할을 섞지 않기 위해, 이 디렉토리는 “실험을 설계하고 기록하는 장소”로 유지합니다.

## 1. 역할

- `models/`: 연구 노트북과 같은 이름의 `.py` 미러
- `results/`: 학습 곡선, CSV, Markdown 보고서
- `research_materials/`: 문헌 리뷰, 설계 메모, 교수님 공유용 브리프
- `images/`: 그림과 시각화 출력
- `scripts/`: 재사용 가능한 보조 유틸리티만 보관

## 2. 연구 규칙

- 새 실험은 반드시 `test/models/*.ipynb` 원본부터 만듭니다.
- 같은 이름의 `.py` 미러를 함께 유지합니다.
- 로컬에서는 무거운 연구 실행을 하지 않고, 실제 결과 산출은 학교 서버 커널이나 승인된 원격 환경에서 수행합니다.
- 공용 프레임워크로 승격할 가치가 있는 로직은 루트 `analysis/`, `contexts/`, `marts/`, `pipelines/`로 옮깁니다.

## 3. 현재 주요 실험

- `1_time_series_test.ipynb`: 기초 시계열 모델 비교
- `2_time_series_advance_test.ipynb`: 고급 시계열 아키텍처 비교
- `3_time_series_multi_ticker_test.ipynb`: KRW 다종목 실험
- `4_text_independent_variable_analysis.ipynb`: 텍스트 독립변수 결합 실험
- `5_optimization_diagnostics_test.ipynb`: objective / architecture별 학습 곡선과 shortcut collapse 진단 실험

## 4. 텍스트 독립변수 실험과의 연결

`4번` 실험은 텍스트가 실제로 설명력 있는 독립변수가 될 수 있는지 비교하는 실험입니다. 여기서는 가격·거래량 계열 변수에 텍스트 이벤트, 감성, 리스크, 규제, 유동성 관련 count를 결합합니다.

이 실험의 질문은 다음과 같습니다.

- 텍스트 맥락이 가격 기반 피처 위에 추가적인 설명력을 주는가
- 특정 뉴스량이나 감성 합계보다, 구조화된 event/context feature가 더 유용한가
- 텍스트를 독립변수로 넣을 때도 naive baseline보다 의미 있는 개선이 있는가

## 5. 최적화 진단 실험

`5번` 실험은 성능 리더보드가 아니라 **학습 곡선 진단 실험**입니다.

핵심 문제의식은 다음과 같습니다.

- 비정상 시계열에서 모델이 정말 구조를 배우는가
- 아니면 loss function이 허용한 쉬운 해로 붕괴하는가
- 이 현상이 단순 기울기 소실이 아니라 `0 수익률 예측`, `lag-1 복사`, `평균 회귀형 flat output` 문제는 아닌가

### 실험 구성

- `objective_probe`: 같은 LSTM 계열에서 target/loss/head만 바꿔 objective의 영향을 관찰
- `architecture_probe`: 같은 objective를 두고 Linear/LSTM/GRU/TCN/Transformer를 비교
- `full_matrix`: 일부 objective와 architecture를 교차해 넓게 스캔

### 주요 진단 지표

- `collapse_score`: 낮을수록 좋음
- `variance_ratio`: 예측 변화량 분산 / 실제 변화량 분산
- `zero_share`: 거의 0에 가까운 수익률을 예측한 비율
- `copy_alignment`: 현재 가격을 복사하는 경향
- `persistence_gap`: naive persistence 대비 개선 여부

즉, 이번 실험의 목적은 “어떤 모델이 가장 잘 맞는가”보다 “어떤 objective가 잘못된 쉬운 해를 덜 허용하는가”를 보는 데 있습니다.

## 6. 실행 예시

실제 실행은 원격/승인 환경에서 진행합니다.

```bash
uv run test/models/4_text_independent_variable_analysis.py --db data/upbit_data.db --table btc_15m_advance
uv run test/models/5_optimization_diagnostics_test.py --suite objective_probe
uv run test/models/5_optimization_diagnostics_test.py --suite architecture_probe --epochs 12
uv run test/models/5_optimization_diagnostics_test.py --suite full_matrix --max-rows 8000 --feature-set text_aware
```

## 7. 결과물 해석

- `results/`의 CSV는 epoch별 학습 곡선과 케이스 요약 표를 담습니다.
- PNG는 학습 곡선과 collapse 진단 지표를 한 번에 보여 줍니다.
- Markdown 보고서는 어떤 objective / architecture 조합이 현재 연구 방향에서 더 안전한지 서술형으로 정리합니다.

## 8. 참고 문서

- [루트 README](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/README.md)
- [AGENTS.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/AGENTS.md)
- [process.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/process.md)
- [history.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/history.md)
- [optimization_context_professor_brief_20260611.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/research_materials/optimization_context_professor_brief_20260611.md)
