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

## 5. 최적화 관련 연구 과제

`5번` 실험은 단순 성능 비교용 리더보드가 아니라, **현재 연구 데이터의 학습 과정에서 어떤 최적화 문제가 생기는지 빠르게 확인하고 이를 어떤 방식으로 통제할지 검토하는 경량 진단 실험**입니다.

이번 연구에서 중요한 점은 금융 시계열이 비정상적이고 잡음이 많다는 이유만으로 학습 실패를 모두 "기울기 소실"로 해석하면 안 된다는 것입니다. 실제로는 모델이 정보를 배우지 못한 것이 아니라, 현재 objective와 prediction head 조합이 허용하는 가장 쉬운 해로 빠져 `0 수익률 근처 예측`, `lag-1 복사`, `평균 회귀형 flat output` 같은 붕괴 패턴을 택할 수 있습니다.

그래서 이 실험은 “어떤 알고리즘이 최고 성능인가”를 고르는 단계가 아니라, **우리 연구 방향에서 어떤 학습 설정은 피해야 하고 어떤 설정은 다음 실험의 기본값으로 삼을 수 있는지 판단하는 진단 단계**로 둡니다. 이후 `/test/models` 아래 연구 실험을 확장할 때, 전처리 이전에 학습 objective 자체가 잘못된 shortcut을 허용하는지 먼저 점검하려는 목적입니다.

또한 지금 목표는 최적화 곡선을 빨리 확인하는 것이므로, 기본 실행은 텍스트 독립변수 전체를 붙인 큰 실험이 아니라 **업비트 시계열의 내생적 변화만 남긴 작은 probe**로 구성합니다. 즉, 이 단계에서는 입력 차원을 줄이고, 겹치는 윈도우 수를 줄이고, 케이스 수도 최소화해서 “붕괴가 있는지 없는지”를 먼저 확인합니다. 빠른 probe에서 이미 collapse가 뚜렷하다면, 독립변수를 더 붙이거나 데이터 크기를 키우는 것은 비용만 늘리고 원인 판단은 오히려 흐릴 수 있다고 봅니다.

### 먼저 확인하려는 연구 질문

- 현재 next-close 또는 return target 설계가 복사형 shortcut을 유도하는가
- Huber, 방향성 penalty, volatility weighting 같은 손실 함수 설계가 collapse를 줄이는가
- 같은 objective를 쓰더라도 Linear, LSTM, GRU 중 어떤 구조가 더 안정적인 학습 곡선을 보이는가

### 실험 구성

- `quick_probe`: 기본 실행. 작은 endogenous feature set과 줄인 window 수로 LSTM 기반 objective 차이만 빠르게 확인
- `objective_probe`: 같은 계열 모델에서 target, loss, prediction head만 바꿔 objective 설계가 붕괴를 유도하는지 확인
- `architecture_probe`: 같은 objective를 둔 상태에서 Linear, LSTM, GRU를 비교해 구조별 학습 안정성을 점검
- `full_matrix`: 줄인 objective와 architecture 조합을 교차시켜 특정 조합에서만 발생하는 붕괴 패턴이 있는지 확인

### 주요 진단 지표

- `collapse_score`: 붕괴 징후를 종합한 지표로 낮을수록 좋음
- `variance_ratio`: 예측 변화량 분산 / 실제 변화량 분산
- `zero_share`: 거의 0에 가까운 수익률을 예측한 비율
- `copy_alignment`: 현재 가격 또는 직전 상태를 복사하는 경향
- `persistence_gap`: naive persistence 대비 개선 여부

정리하면, 이번 실험의 목적은 “무슨 모델이 제일 잘 맞는가”보다 “어떤 objective와 구조 조합이 연구용 학습 과정에서 잘못된 쉬운 해로 무너지는가”를 먼저 가려내는 데 있습니다.

이 때문에 `5번` 실험은 **빠르게 실패 양상을 찾는 1차 필터**로 이해하는 것이 맞고, 독립변수 추가 및 본격 예측 실험은 그 다음 단계에서 진행하는 것이 더 합리적입니다.

## 6. 실행 예시

실제 실행은 원격/승인 환경에서 진행합니다.

```bash
uv run test/models/4_text_independent_variable_analysis.py --db data/upbit_data.db --table btc_15m_advance
uv run test/models/5_optimization_diagnostics_test.py
uv run test/models/5_optimization_diagnostics_test.py --suite objective_probe
uv run test/models/5_optimization_diagnostics_test.py --suite architecture_probe --max-windows 768 --epochs 6
uv run test/models/5_optimization_diagnostics_test.py --suite full_matrix --feature-set market_only --max-rows 4000
```

## 7. 결과물 해석

- 기본 원칙은 `ipynb` 출력 셀을 1차 결과물로 보는 것입니다.
- 서버 커널로 실행한 뒤 노트북에는 Markdown, 표, figure를 모두 출력하고, 로컬 Codex는 그 출력 셀과 이미지를 파싱해서 보고서를 작성합니다.
- `test/scripts/extract_notebook_images.py`는 노트북 출력 이미지 추출용 보조 유틸리티입니다.
- `results/`의 CSV는 필요할 때만 저장하는 보조 산출물입니다.
- PNG는 학습 곡선, 모델별 curve, collapse 진단 지표를 보여 줍니다.
- Markdown 보고서는 어떤 objective / architecture 조합이 현재 연구 방향에서 더 안전한지, 그리고 그 이유가 무엇인지 서술형으로 정리합니다.
- 보고서는 먼저 방법론과 지표 정의를 설명하고, 그다음 데이터 조건과 기초통계량, 마지막으로 결과를 해석하는 순서를 따릅니다.

## 8. 참고 문서

- [루트 README](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/README.md)
- [AGENTS.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/AGENTS.md)
- [process.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/process.md)
- [history.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/history.md)
- [optimization_context_professor_brief_20260611.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/research_materials/optimization_context_professor_brief_20260611.md)
