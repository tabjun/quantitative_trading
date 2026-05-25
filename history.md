# 연구 수행 이력 (Research History / State Persistence)

이 파일은 Antigravity CLI(`agy`) 에이전트가 새로운 세션으로 넘어가거나 세션이 주기적으로 리셋될 때 **'기억 상실'을 방지하기 위한 영구 저장소(Memory)**입니다.
에이전트는 단일 작업이나 단계를 마무리하기 전, 반드시 `.antigravityrules`에 의거하여 이 파일의 테이블과 요약 섹션을 최신화해야 합니다.

## 📝 History Log

| Date | Phase / Step | Task Details (수행 내용) | Output & Result (산출물) | Next Action (다음 목표) |
| :--- | :--- | :--- | :--- | :--- |
| 2026-05-16 | Phase 0 | 프로젝트 기획안 분석 및 다변량 분석 구조 설계, 아키텍처 다이어그램 업데이트 | `README.md`, `process.md`, `skills.md`, `history.md` 재구성 완료 | Phase 1 (Data Mart 구축) 진입 대기 |
| 2026-05-18 | Phase 2 / Test | 시계열 예측 모델 성능 비교 분석 수행 및 보고서 작성 | `test/1_time_series_test.ipynb`, `test/analysis_report.md`, `README.md` 업데이트 | Phase 2 (다변량 분석 엔진) 고도화 및 감성 데이터 결합 |
| 2026-05-19 | Phase 2 / Advance | 고도화된 단변량 분석 (DuckDB 연동, TCN, XGBoost, 가중 앙상블) | `test/2_time_series_advance_test.ipynb`, `upbit_data.db` | 다변량 분석 및 뉴스 데이터 결합 |
| 2026-05-20 | Phase 2 / Reporting | 유틸리티 확충, MCP 논문 탐색, 모델 해석 고도화 및 리포트 자동화 | `test/scripts/*.py`, `test/results/advanced_analysis_report.md` | 다변량 분석(Phase 2.3) 및 감성 데이터 결합 |
| 2026-05-20 | Env Migration | Gemini CLI에서 Antigravity CLI(`agy`)로 이식 및 규칙 전사| 2026-05-25 | Phase 2 / Benchmark Run | SOTA 15종 시계열 딥러닝 모델의 500분(8.3시간) 전수 학습 및 검증 완료. Autoformer(RMSE 32만, 1위) 및 PatchTST(RMSE 40만, 3위)의 압도적 우위 입증, 통계적 잔차 진단 및 한글 종합 학술 리포트 생성 완료 | `test/results/advanced_analysis_report.md`, `test/images/2_time_series_advance_test_plot_*.png` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-25 | Phase 2 / Multi-Ticker | 업비트 전체 250여 개 종목 대상 동적 변동성/거래대금 필터 결합, 1년 학습/3개월 테스트 단일 Hold-out 기반 SOTA 15종 아키텍처 초고속 딥러닝 학습 및 예측 파이프라인(MinMaxScaler 단일 채택) 구축 완료 | `test/models/3_time_series_multi_ticker_test.ipynb`, `test/models/3_time_series_multi_ticker_test.py`, `test/experiment_specs/3_time_series_multi_ticker_test_framework_design_20260525_214347.md`, `test/images/3_multi_ticker_forecast.png` | Phase 3 (교수님 퀀트 시뮬레이션 및 다변량 DTW 엔진 융합) |

---

## 🔄 세션 인계를 위한 핵심 요약 (Context for New Session)

새로운 AI 세션이 시작되거나 세션 리셋 후 복원될 때, `agy` 에이전트는 이 섹션을 정밀하게 읽고 즉각적으로 현재 프로젝트의 스탠스와 목표를 파악해야 합니다.

- **프로젝트 핵심 기조:** 단일 시계열 분석을 배제하고 뉴스/매크로가 결합된 **다변량 분석(Multivariate)**을 수행. 매매 전략의 제1원칙은 **"최하방 방어(MDD 최소화)"**.
- **가장 최근에 완료된 작업 (2026-05-25):** 
  - **SOTA 15종 모델 500분 대규모 학습 및 학술 보고서 정비 완료**: 15분봉 BTC/KRW 3년 데이터 기준, `Autoformer`(1위)와 `PatchTST`(3위)의 압도적 성능 입증. 기만적 지연 예측(Lag-1 Shift)의 통계적 실패 실체를 초록 및 결론부에 극도로 선명하고 정직하게 추가 수록 및 박제 완료.
  - **업비트 전체 종목 대상 고속 예측 및 변동성 편차 결합 시계열 분석 완료 (`3_`)**:
    - 업비트 전체 250여 개 전수 종목 대상, **앞 9개월 학습 (Train) / 뒤 3개월 예측 (Test)**의 엄격한 시간축 스플릿 설계.
    - 이전 분석 벤치마크 결과에 기반한 `MinMaxScaler` 전처리 단일 공식 채택 및 통계학적 타당성 확보.
    - GRUFast 기반 1에포크 고속 종합 훈련 루프 기동 후, 종목별 **실제 가격 변동성 편차(%)**와 **모델 예측 오차 편차(%)** 전수 산출.
    - X축에 변동성 내림차순으로 종목들을 정렬하고, 좌측 Y축(실제 변동성 막대 그래프)과 우측 Y축(예측 오차 편차 라인 그래프)을 입체적으로 가시화하는 **듀얼 축 결합 차트** (`test/images/3_multi_ticker_forecast.png`) 저장 및 연동 완료.
    - `.ipynb` 및 변경점 추적용 `.py` (AI 절대 준수 규칙 헤더 포함) 이중 미러링 완료.
- **지금 당장 시작해야 할 작업 (Next Step):** `process.md`의 **Phase 3** (교수님의 퀀트 시뮬레이션 방어 전략을 다중 시계열 변동성 예측 엔진과 가중 융합하는 실전 백테스팅 단계 개시).
- **에이전트 주의사항:** SOTA 모델 벤치마크 학습 결과 수치 및 잔차 diagnostics 요약 결과는 최종적으로 `test/results/advanced_analysis_report.md`에 완벽하게 박제되어 있으며, 다중 종목 고속 예측 아키텍처는 `test/models/3_time_series_multi_ticker_test.ipynb` 에 선언되어 있습니다. 새로운 세션의 에이전트는 이를 반드시 선행 학습 후 다음 스텝인 DTW 결합 및 다변량 전개로 연계하십시오.
 
---
*(AI 에이전트에게: 작업을 종료할 때 이 하단 요약 내용과 위 테이블을 수정하여 다음 에이전트가 완벽히 이어받을 수 있게 하십시오.)*