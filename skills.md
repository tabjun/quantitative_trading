# 필요 기술 및 분석 기법 (Skills & Methods)

본 프로젝트의 다중 에이전트들이 수행하는 핵심 머신러닝/딥러닝 및 데이터 분석 기법입니다.

## 1. Dynamic Time Warping (DTW) - 시계열 패턴 매칭
- **목적:** 길이가 다르거나 속도가 다른 두 시계열 데이터 간의 유사성을 측정합니다.
- **활용:** 현재의 2주일간의 하락 패턴이, 과거 2018년 미중 무역분쟁 당시의 1개월 하락 패턴과 얼마나 구조적으로 유사한지 거리를 계산합니다. 유클리디안 거리(Euclidean Distance)의 한계를 극복하여 파동의 '형태' 자체를 비교합니다.
- **사용 라이브러리:** `fastdtw`, `numpy`, `scipy`

## 2. Text Embedding & Cosine Similarity - 문맥 유사도 분석
- **목적:** 뉴스 기사나 거시 경제 이슈의 텍스트가 의미적으로 얼마나 유사한지 비교합니다.
- **활용:** "중동 지정학적 리스크 고조"와 "걸프전 발발"이라는 텍스트를 벡터 공간(Vector Space)으로 변환하여, 단어가 달라도 문맥적 유사성이 높다는 것을 AI가 인지하게 합니다.
- **사용 기술:** OpenAI Embeddings API (또는 Sentence-Transformers), Cosine Similarity

## 3. Multi-Agent Orchestration (Supervisor Pattern)
- **목적:** 복잡한 분석 로직을 분산 처리하고, 각 에이전트가 자신의 전문 분야(차트 분석, 뉴스 분석, 리포트 작성)에 집중하게 합니다.
- **활용:** 객체 지향적 설계를 통해 `Supervisor`가 `SituationAnalyzer`, `SimilarityMatcher`, `ReportGenerator`를 순차적으로 호출하며 파이프라인의 입출력을 관리합니다.

## 4. Prompt Engineering (XAI 기반 설명)
- **목적:** 단순한 수치적 일치율(예: DTW 거리 0.15)을 투자자가 직관적으로 이해할 수 있는 언어로 번역합니다.
- **활용:** "현재 차트는 2022년 금리 인상 충격 당시와 85% 일치하는 흐름을 보이며, 당시 반등의 트리거는 인플레이션 지수 꺾임이었습니다"라는 식의 설명 가능한 AI(Explainable AI) 리포트를 작성합니다.
