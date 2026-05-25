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
| 2026-05-20 | Env Migration | Gemini CLI에서 Antigravity CLI(`agy`)로 이식 및 규칙 전사화 완료 | `.antigravityrules` 생성, `process.md`, `history.md`, `skills.md`, `README.md` 개편 완료 | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-24 | Phase 2 / Step 2.0 (Advance) | 지연 예측(Lagging) 극복 및 시계열 전처리 다변화(차분, 로그수익률, Z-score), 2년 훈련/1년 예측 strict 분할, 잔차 통계 진단(DW, JB, Ljung-Box) 및 한글 레포트 생성 모듈, 5종 고화질(300 DPI) 로컬 이미지 저장 아카이브 구축, PyTorch LSTM/GRU `input_size` 차원 불일치(Expected 1, got 60) shape 문제 해결 | `test/models/2_time_series_advance_test.ipynb`, `test/models/2_time_series_advance_test.py`, `test/images/*` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-25 | Phase 2 / Framework Doc | 6대 시계열 전처리 수식, 역변환 매커니즘, 5대 고도화 손실함수, 15종 알고리즘 상세, 8대 종합 평가지표(MASE, DTW 등) 및 3대 오차 검정법을 총망라한 종합 실험 프레임워크 명세서 구축 완료. 프로젝트 내 4개 유틸리티 스크립트(create_advance_nb.py 등)에 상세 독스트링 기술 설명 전면 반영 | `test/experiment_specs/2_time_series_advance_test_framework_design_20260525_050836.md` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-25 | Phase 2 / Benchmark Run | SOTA 15종 시계열 딥러닝 모델의 500분(8.3시간) 전수 학습 및 검증 완료. Autoformer(RMSE 32만, 1위) 및 PatchTST(RMSE 40만, 3위)의 압도적 우위 입증, 통계적 잔차 진단 및 한글 종합 학술 리포트 생성 완료 | `test/results/advanced_analysis_report.md`, `test/images/2_time_series_advance_test_plot_*.png` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |

---

## 🔄 세션 인계를 위한 핵심 요약 (Context for New Session)

새로운 AI 세션이 시작되거나 세션 리셋 후 복원될 때, `agy` 에이전트는 이 섹션을 정밀하게 읽고 즉각적으로 현재 프로젝트의 스탠스와 목표를 파악해야 합니다.

- **프로젝트 핵심 기조:** 단일 시계열 분석을 배제하고 뉴스/매크로가 결합된 **다변량 분석(Multivariate)**을 수행. 매매 전략의 제1원칙은 **"최하방 방어(MDD 최소화)"**.
- **가장 최근에 완료된 작업 (2026-05-25):** 
  - **SOTA 15종 모델 500분 대규모 학습 완료**: 15분봉 BTC/KRW 3년 데이터 기준, `Autoformer` (RMSE **320,017 KRW**, 1위)와 `PatchTST` (RMSE **404,392 KRW**, 3위)가 기존 LSTM(53만)/GRU(61만) 및 순방향 트랜스포머 변형군 대비 압도적 오차 하락을 실증함.
  - **예측 지연(Lagging) 극복 검증**: 1차 차분 및 로그 수익률 전처리 파이프라인(`PreprocessingPipeline`)의 적용으로 모델들이 직전 시간 종가를 그대로 복사해 오던 lagging 현상을 원천 차단하고 점진적으로 우하향하는 정석적이고 안정적인 손실 학습 곡선을 확보함.
  - **학술적 잔차 통계 검정 및 보고서 아카이빙**: Durbin-Watson(자기상관 `1.48` 수준으로 양호하지만 미세 모멘텀 잔존), Jarque-Bera(p < 0.05로 오차의 Fat Tail 비정규성 입증되어 손절 -2% 등 자금 관리 룰 필수성 수학적 입증) 등 3대 잔차 검정 결과를 집대성한 논문 수준의 리포트(`test/results/advanced_analysis_report.md`)와 300 DPI 초고화질 시각화 플롯 5종 생성 완료.
- **지금 당장 시작해야 할 작업 (Next Step):** `process.md`의 **Phase 2, Step 2.1** (시계열 패턴 매칭 DTW 구현). Durbin-Watson 진단에서 제기된 미세한 추세 이탈 오차를 보강해 주기 위해, `fastdtw` 등을 활용하여 현재 하락/상승 파동 국면과 가장 닮은 과거 폭락/폭등 사례를 고속 색출하여 딥러닝 보정항으로 탑재하는 패턴 매칭 엔진을 개발할 것.
- **에이전트 주의사항:** SOTA 모델 벤치마크 학습 결과 수치 및 잔차 diagnostics 요약 결과는 최종적으로 `test/results/advanced_analysis_report.md`에 완벽하게 박제되어 있으므로, 새로운 세션의 에이전트는 이를 반드시 선행 학습 후 다음 스텝인 DTW 결합 및 다변량 전개로 연계하십시오.
 
---
*(AI 에이전트에게: 작업을 종료할 때 이 하단 요약 내용과 위 테이블을 수정하여 다음 에이전트가 완벽히 이어받을 수 있게 하십시오.)*

